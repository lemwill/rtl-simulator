# SIMD Acceleration for Repeated Module Simulation

## 1. The Opportunity

AI accelerators, NoCs, multi-core processors, DSP arrays, and crypto engines all contain
large arrays of **identical processing elements (PEs)**. A 16×16 systolic array has 256
identical PEs. An AI chip might have thousands. Traditional simulators evaluate each instance
sequentially -- a massive waste when the logic is identical and only the data differs.

This is textbook SIMD: **Same Instruction, Multiple Data**.

## 2. How It Works

### 2.1 Data Layout Transformation: AoS → SoA

The prerequisite is transforming signal storage from Array-of-Structures to Structure-of-Arrays.

**AoS (how designers think, how Verilator stores signals):**
```
Memory: [PE0.a, PE0.b, PE0.acc, PE0.valid | PE1.a, PE1.b, PE1.acc, PE1.valid | ...]
```
Signals from one instance are contiguous. Bad for SIMD -- loading `a` from 16 PEs
requires 16 scattered loads.

**SoA (what SIMD needs):**
```
Memory: [PE0.a, PE1.a, PE2.a, ... PE255.a | PE0.b, PE1.b, ... PE255.b | ...]
```
The same signal across all instances is contiguous. Loading `a` from 16 PEs is a
single aligned SIMD load.

### 2.2 Signal Width → SIMD Throughput

| Signal Width | AVX2 (256-bit) | AVX-512 (512-bit) | Instances/instruction |
|-------------|----------------|--------------------|-----------------------|
| 1-bit       | 256 lanes      | 512 lanes          | 256 / 512             |
| 8-bit       | 32 lanes       | 64 lanes           | 32 / 64               |
| 16-bit      | 16 lanes       | 32 lanes           | 16 / 32               |
| 32-bit      | 8 lanes        | 16 lanes           | 8 / 16                |
| 64-bit      | 4 lanes        | 8 lanes            | 4 / 8                 |

For 1-bit control signals, **256 PEs fit in a single AVX2 register**. One AND instruction
evaluates 256 PEs simultaneously.

For int8 values (common in quantized AI): **64 PEs per AVX-512 instruction**.

### 2.3 Mapping RTL Operations to SIMD Instructions

#### Bitwise Logic (AND, OR, XOR, NOT)
Maps perfectly. These are the bread and butter of RTL simulation.

```cpp
// valid_out = valid_in & enable   (for 16 PEs, 32-bit packed)
__m512i valid_out = _mm512_and_epi32(valid_in, enable);

// For 1-bit signals packed into a single register (512 PEs):
__m512i valid_out = _mm512_and_si512(valid_in, enable);
```

**Cost**: 1 cycle latency, 1-per-cycle throughput. Identical to scalar.

#### Arithmetic (ADD, SUB, MUL)
```cpp
// acc += a * b   (16 × 32-bit PEs simultaneously)
__m512i prod = _mm512_mullo_epi32(a_vec, b_vec);
acc_vec = _mm512_add_epi32(acc_vec, prod);

// For int8 MAC with AVX-512 VNNI (hardware acceleration for AI inference):
// Processes 64 × int8 multiply-accumulates in ONE instruction
acc_vec = _mm512_dpbusd_epi32(acc_vec, a_bytes, b_bytes);
```

The VNNI instruction is remarkable: it does exactly what a systolic array PE does
(int8 multiply-accumulate), and it processes 64 PEs per instruction. This means
**simulating an AI MAC array using the same instructions that execute AI inference**.

#### Multiplexers (the key challenge)
```cpp
// result = sel ? a : b   (16 PEs, 32-bit)
__mmask16 mask = _mm512_cmpeq_epi32_mask(sel_vec, _mm512_set1_epi32(1));
__m512i result = _mm512_mask_blend_epi32(mask, b_vec, a_vec);

// 4-to-1 mux: chain two blends, or use a lookup table approach
```

AVX-512's mask registers (`__mmask16`) make conditional operations very efficient.
With AVX2, use `_mm256_blendv_epi8` (slightly less flexible).

#### Shifts and Rotates
```cpp
// shift_out = data >> shift_amount  (16 PEs, each shifting independently)
__m512i result = _mm512_srlv_epi32(data_vec, shift_vec);  // variable shift

// Same shift amount for all PEs (common: barrel shifter with shared control):
__m512i result = _mm512_srli_epi32(data_vec, 4);  // shift all right by 4
```

#### Comparisons
```cpp
// is_zero = (acc == 0)  for 16 PEs
__mmask16 is_zero = _mm512_cmpeq_epi32_mask(acc_vec, _mm512_setzero_si512());

// saturating_add: clamp result to MAX if overflow
__m512i sum = _mm512_adds_epu16(a_vec, b_vec);  // unsigned saturating add
```

### 2.4 Inter-Instance Communication (Systolic Data Flow)

This is the most interesting part. In a systolic array, data flows between neighbors:

```
PE[0] → PE[1] → PE[2] → ... → PE[15]   (1D shift)

PE[0][0] → PE[0][1]    (horizontal flow)
    ↓          ↓
PE[1][0] → PE[1][1]    (vertical flow)
```

SIMD has **shuffle/permute** instructions designed for exactly this kind of data movement.

#### 1D Shift (PE[i].out → PE[i+1].in)

```cpp
// Shift right by 1 position within a 16-element AVX-512 register
// Elements: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
// Result:   [X,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  (X = from previous chunk)

__m512i idx = _mm512_set_epi32(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0);
__m512i shifted = _mm512_permutexvar_epi32(idx, data);
// Then insert the boundary element from the previous register chunk
shifted = _mm512_mask_set1_epi32(shifted, 0x0001, boundary_value);
```

For a 16×16 systolic array with 32-bit PEs:
- Each row = 16 PEs × 32 bits = 512 bits = **exactly 1 AVX-512 register**
- Horizontal shift = 1 permute instruction per row = **16 instructions for the whole array**
- Vertical shift = load from next row's SoA array = **regular memory access**

This is incredibly efficient.

#### 2D Mesh Communication

```cpp
// For a 16×16 array, rows packed in AVX-512 registers:
// Horizontal neighbor: permute within register (1 instruction)
// Vertical neighbor: load from adjacent row array (1 load)
// Diagonal: combine permute + cross-row load

for (int row = 0; row < 16; row++) {
    __m512i h_in = shift_right(pe_row[row]);           // 1 permute
    __m512i v_in = _mm512_load_epi32(&pe_out[(row-1) * 16]); // 1 load
    // ... evaluate PE logic with these inputs ...
}
// Total: ~16 rows × ~10 SIMD ops = ~160 instructions for 256 PEs
```

#### Cross-Register Boundary Handling

When PE arrays are larger than one SIMD register (e.g., 64 PEs in a row with 32-bit data
= 4 AVX-512 registers), data must flow between registers at boundaries:

```cpp
// Extract last element of register N, insert as first element of register N+1
uint32_t boundary = _mm512_cvtsi512_si32(
    _mm512_permutexvar_epi32(_mm512_set1_epi32(15), reg_n));
reg_n_plus_1 = _mm512_mask_set1_epi32(
    shift_right(reg_n_plus_1), 0x0001, boundary);
```

This is a few extra instructions at chunk boundaries -- negligible overhead.

### 2.5 Control Flow Divergence (Non-Uniform PE States)

Not all PEs are always doing the same thing. Some may be:
- Draining (processing last elements)
- Idle (waiting for data)
- In a different FSM state

AVX-512's **per-element masking** handles this elegantly:

```cpp
// Only update PEs that are in COMPUTE state
__mmask16 computing = _mm512_cmpeq_epi32_mask(state_vec, COMPUTE_STATE);

// Masked multiply-accumulate: only active PEs update
acc_vec = _mm512_mask_add_epi32(
    acc_vec,                                    // unchanged lanes keep old value
    computing,                                  // only update lanes where mask=1
    acc_vec,
    _mm512_mullo_epi32(a_vec, b_vec)
);
```

**Cost**: masked operations have the same throughput as unmasked on modern CPUs.
No wasted work for inactive PEs.

## 3. Compile-Time Detection and Transformation

### 3.1 Detection Algorithm

```
DETECT_SIMD_CANDIDATES(elaborated_design):
  for each module_type M in design:
    instances = find_all_instances(M)
    if len(instances) < SIMD_THRESHOLD:  // e.g., 8
      continue

    // Check structural uniformity
    if not all_same_parameters(instances):
      continue

    // Analyze connectivity
    classify_ports(instances):
      shared_ports:        same source for all instances (clk, rst, config)
      per_instance_ports:  unique source per instance (data inputs)
      inter_instance_ports: connected to neighboring instances (systolic flow)
      external_ports:      connected to non-array logic

    // Check SIMD viability
    score = compute_simd_benefit(M, instances)
    if score > threshold:
      mark_for_simd(M, instances, connectivity_info)
```

### 3.2 Connectivity Pattern Classification

| Pattern | Example | SIMD Strategy |
|---------|---------|---------------|
| **Broadcast** | clk, rst, config regs | Load once, use `_mm512_set1_*` |
| **Independent** | each PE has unique input from memory | Standard SoA load |
| **Shift** | systolic left/right/up/down | Permute instructions |
| **Reduction** | all PE outputs OR'd together | Horizontal reduction ops |
| **Scatter/Gather** | irregular routing | `_mm512_i32gather_*` |

### 3.3 Code Generation Strategy

For each SIMD-candidate module array:

1. **Allocate SoA arrays** (aligned to 64 bytes for AVX-512):
   ```cpp
   alignas(64) uint32_t pe_a[NUM_PES];
   alignas(64) uint32_t pe_b[NUM_PES];
   alignas(64) uint32_t pe_acc[NUM_PES];
   ```

2. **Generate vectorized eval function**:
   ```cpp
   void eval_pe_array() {
       for (int i = 0; i < NUM_PES; i += SIMD_WIDTH) {
           __m512i a   = _mm512_load_epi32(&pe_a[i]);
           __m512i b   = _mm512_load_epi32(&pe_b[i]);
           __m512i acc = _mm512_load_epi32(&pe_acc[i]);
           // ... vectorized logic ...
           _mm512_store_epi32(&pe_acc[i], acc);
       }
   }
   ```

3. **Handle boundary logic** (first/last PE, edge of array) with masking.

4. **Generate inter-instance communication** as permute operations.

### 3.4 LLVM IR Generation

Instead of hand-writing intrinsics, emit LLVM IR with vector types:

```llvm
; Vectorized PE evaluation (16 × i32)
define void @eval_pe_array(ptr %pe_a, ptr %pe_b, ptr %pe_acc, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %a_ptr = getelementptr i32, ptr %pe_a, i64 %i
  %b_ptr = getelementptr i32, ptr %pe_b, i64 %i
  %acc_ptr = getelementptr i32, ptr %pe_acc, i64 %i

  %a = load <16 x i32>, ptr %a_ptr, align 64
  %b = load <16 x i32>, ptr %b_ptr, align 64
  %acc = load <16 x i32>, ptr %acc_ptr, align 64

  %prod = mul <16 x i32> %a, %b
  %new_acc = add <16 x i32> %acc, %prod

  store <16 x i32> %new_acc, ptr %acc_ptr, align 64
  %i.next = add i64 %i, 16
  %done = icmp uge i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
```

LLVM will lower this to the best available SIMD instructions for the target CPU
(AVX-512, AVX2, NEON, etc.) automatically. This gives us **portable SIMD** through
LLVM's vector types rather than architecture-specific intrinsics.

## 4. Performance Estimates

### 4.1 Systolic Array Example: 16×16, INT8 MAC PEs

Each PE per cycle: load a (8-bit), load b (8-bit), multiply-accumulate into acc (32-bit),
shift a right, shift b down. ~10 operations per PE.

**Sequential simulation:**
- 256 PEs × 10 ops × ~1.5 ns/op (with cache misses) = ~3,840 ns/cycle

**SIMD (AVX-512, 16 × 32-bit lanes):**
- 256 PEs / 16 lanes = 16 chunks
- 16 chunks × 10 SIMD ops × ~0.5 ns/op = ~80 ns/cycle
- **Speedup: ~48×** for the PE array

**SIMD (AVX-512 VNNI, for the MAC operation specifically):**
- The `vpdpbusd` instruction does 4 × int8 MAC per 32-bit lane, 16 lanes
- 256 PEs / (16 lanes × 4 per lane) = 4 instructions for the MAC step
- Even more dramatic speedup for the multiply-accumulate path

### 4.2 Whole-Design Impact

In a typical AI accelerator:
- PE array: 60-80% of simulation time
- Control logic + memory interfaces: 20-40%

If PE array gets 16-48× speedup from SIMD, and control logic stays 1×:

| PE Array % | SIMD Speedup | Whole-Design Speedup |
|-----------|--------------|---------------------|
| 60%       | 16×          | 3.5×                |
| 60%       | 48×          | 4.5×                |
| 80%       | 16×          | 4.5×                |
| 80%       | 48×          | 8.5×                |

And this stacks with other Surge optimizations (LLVM JIT, multi-threading, etc.).

### 4.3 Combined with Multi-Threading

The SIMD-optimized PE array can itself be split across threads:
- Thread 0: PEs 0-127 (SIMD-vectorized)
- Thread 1: PEs 128-255 (SIMD-vectorized)
- Thread 2: Control logic (scalar)

SIMD × multi-threading = multiplicative speedup.

## 5. Advanced Techniques

### 5.1 Two-State SIMD Optimization for 1-bit Signals

For 1-bit control signals (valid, ready, stall, flush), pack all instances into a single
machine word. With AVX-512, **512 PEs' control signals in one register**:

```cpp
// 512 PEs' valid bits in one register
__m512i valid_all = _mm512_load_si512(&pe_valid_bits);
__m512i enable_all = _mm512_load_si512(&pe_enable_bits);
__m512i active = _mm512_and_si512(valid_all, enable_all);
// 512 PEs evaluated in 1 instruction!
```

### 5.2 Four-State SIMD

For opt-in four-state (X/Z support), each signal needs 2 bits (value + control).
With SoA layout, store value and control in separate arrays:

```cpp
// Four-state AND for 16 PEs:
// val_out = (val_a & val_b) & ~(ctrl_a | ctrl_b)  |  (val_a & val_b) & (ctrl_a | ctrl_b)
// ctrl_out = ctrl_a | ctrl_b | (~val_a & ctrl_b) | (ctrl_a & ~val_b)
// (Simplified; actual X propagation rules are more complex)

__m512i va = _mm512_load_epi32(&val_a[i]);
__m512i ca = _mm512_load_epi32(&ctrl_a[i]);
__m512i vb = _mm512_load_epi32(&val_b[i]);
__m512i cb = _mm512_load_epi32(&ctrl_b[i]);

__m512i val_out = _mm512_and_epi32(va, vb);
__m512i ctrl_out = _mm512_or_epi32(ca, cb);
// ... (full truth table implementation)
```

Four-state SIMD is 2× the operations of two-state SIMD, but still 8-32× faster
than sequential four-state scalar simulation.

### 5.3 Waveform Tracing with SIMD

Change detection (for selective waveform dumping) benefits from SIMD:

```cpp
// Compare old and new state for 16 PEs at once
__m512i old_acc = _mm512_load_epi32(&pe_acc_prev[i]);
__m512i new_acc = _mm512_load_epi32(&pe_acc[i]);
__mmask16 changed = _mm512_cmpneq_epi32_mask(old_acc, new_acc);

if (changed != 0) {
    // Only dump signals for PEs that actually changed
    // Use _mm512_mask_compressstoreu_epi32 to gather changed values
}
```

### 5.4 Parameterized Instances (Partially Different)

Not all "similar" modules are identical. Some may have different parameters
(e.g., PE[i] has DEPTH=i for a triangular array). Strategies:

1. **Group by parameter**: If there are only K distinct parameter sets,
   create K SIMD groups
2. **Superset simulation**: Simulate all instances as the largest variant,
   mask out irrelevant operations for smaller variants
3. **Fall back to scalar**: If too heterogeneous, don't vectorize

## 6. Applicability Beyond AI Accelerators

| Design Pattern | Example | Instances | SIMD Benefit |
|---------------|---------|-----------|-------------|
| Systolic array | TPU, Gemmini, NVDLA | 64-4096 | Excellent |
| SRAM banks | Cache arrays, scratchpads | 4-64 | Good |
| NoC routers | Mesh/torus interconnect | 16-256 | Good |
| CPU cores | Multi-core processor | 2-64 | Good (large modules) |
| SerDes lanes | PCIe/USB PHY | 4-32 | Good |
| Filter taps | FIR/IIR DSP | 8-256 | Excellent |
| FFT butterflies | FFT/NTT accelerator | 16-1024 | Excellent |
| Hash units | SHA/AES parallel | 4-16 | Good |
| GPIO pins | Pin controllers | 8-128 | Moderate |
| Register file entries | Reg file, FIFO entries | 16-256 | Good |

## 7. Implementation Plan for Surge

### Phase 1: Detection (v0.2)
- Walk elaborated design, identify module arrays
- Classify connectivity patterns
- Report SIMD candidates to the user (`--simd-report`)

### Phase 2: SoA Transformation (v0.2)
- Transform signal storage layout for identified arrays
- Ensure proper alignment for SIMD loads/stores
- Handle boundary between vectorized and scalar regions

### Phase 3: LLVM Vector IR Emission (v0.3)
- Emit LLVM IR with `<N x i32>` vector types for vectorizable operations
- Let LLVM lower to target-specific SIMD (AVX2/AVX-512/NEON)
- Handle inter-instance communication via shuffle vectors

### Phase 4: Advanced Optimizations (v0.4+)
- AVX-512 masking for divergent control flow
- VNNI/AMX instructions for int8 MAC operations
- Cross-register boundary optimization
- Four-state SIMD

## 8. Key SIMD Instruction Reference

### AVX2 (256-bit, available on all modern x86)
| Operation | Instruction | Latency | Throughput |
|-----------|-------------|---------|------------|
| 32-bit AND | `vpand` | 1 | 0.33 |
| 32-bit ADD | `vpaddd` | 1 | 0.5 |
| 32-bit MUL | `vpmulld` | 10 | 1 |
| Blend/select | `vpblendvb` | 2 | 1 |
| Permute32 | `vpermd` | 3 | 1 |
| Compare | `vpcmpeqd` | 1 | 0.5 |

### AVX-512 (512-bit, Intel Xeon, recent AMD)
| Operation | Instruction | Latency | Throughput |
|-----------|-------------|---------|------------|
| 32-bit AND | `vpandd` | 1 | 0.5 |
| 32-bit ADD | `vpaddd` | 1 | 0.5 |
| 32-bit MUL | `vpmulld` | 10 | 1 |
| Masked blend | `vpblendmd` | 1 | 0.5 |
| Permute32 | `vpermd` | 3 | 1 |
| Compress | `vpcompressd` | 3 | 1 |
| Gather | `vpgatherdd` | varies | varies |
| VNNI int8 MAC | `vpdpbusd` | 5 | 1 |

### ARM NEON (128-bit, all Apple Silicon, ARM servers)
| Operation | Instruction | Width |
|-----------|-------------|-------|
| 32-bit AND | `AND v.4S` | 4 lanes |
| 32-bit ADD | `ADD v.4S` | 4 lanes |
| 32-bit MUL | `MUL v.4S` | 4 lanes |
| Select | `BSL` | 4 lanes |
| Table lookup | `TBL` | flexible |

Note: Apple M-series has excellent NEON throughput. ARM SVE/SVE2 (on server ARM)
provides variable-length vectors (128-2048 bit) similar to AVX-512.
