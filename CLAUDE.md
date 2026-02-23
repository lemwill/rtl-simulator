# Surge - The Fastest Open-Source RTL Simulator

## What To Do Next

**Sprint 12 complete.** Block value tracking, logical NOT fix, concat LHS, inc/dec ops, CRC-32 benchmark.

Next priorities:
1. **While/repeat loops**: Procedural loops beyond for-loops.
2. **Struct types**: Packed structs with field access.
3. **Memory (2D arrays)**: Larger memories, block RAM modeling.
4. **Multi-driven signal resolution**: Arbitrate multiple drivers to same signal.

### Project Structure
```
surge/
  CMakeLists.txt
  src/
    main.cpp              # CLI entry point
    ir/                   # Surge intermediate representation
      ir.h                # IR node types (Module, Signal, Assignment, etc.)
      builder.cpp         # slang AST → Surge IR lowering
    codegen/              # LLVM code generation
      codegen.h
      codegen.cpp         # Surge IR → LLVM IR → JIT native code
    sim/                  # Simulation runtime
      runtime.h
      runtime.cpp         # Clock loop, signal state, eval dispatch
    trace/                # Waveform output
      vcd_writer.h
      vcd_writer.cpp
  include/surge/         # Public headers (for testbench API)
    surge.h
  tests/                 # Test designs
    counter.sv
    adder.sv
    lfsr.sv
    fsm.sv
    alu_regfile.sv
    alu_regfile_stim.sv
    pipeline_datapath.sv
    counter_hier.sv
    riscv_pipeline.sv
    barrel_shifter.sv
    fifo.sv
    crc32.sv
    param_adder.sv
    generate_chain.sv
    enum_fsm.sv
  bench/
    run_bench.sh            # Surge vs Verilator benchmark
```

### Dependencies
All dependencies must be managed by the build system (CMake FetchContent or system packages). No manual installs or git submodules.
- **slang**: FetchContent from GitHub master, MIT license. SV frontend.
- **LLVM 17+**: system install via brew/apt. JIT backend (OrcJIT). Linked via llvm-config (not CMake targets, which inject problematic -isystem flags on macOS).
- **C++20 compiler**: Homebrew clang on macOS (Apple Clang lacks `<source_location>`), GCC 12+ on Linux.
- **CMake 3.20+**: Build system.

---

## Key Decisions

- **License**: Apache 2.0
- **Language**: C++20
- **Frontend**: [slang](https://github.com/MikePopoloski/slang) (MIT, C++17) for IEEE 1800-2017 SystemVerilog
- **Backend**: LLVM JIT (slang AST → Surge IR → LLVM IR → native code via OrcJIT)
- **Simulation model**: Cycle-based first. Hybrid adaptive (cycle-based + event-driven) later.
- **State model**: Two-state (0/1) default. Opt-in four-state (X/Z) later.
- **Parallelism**: Single-threaded first. Lock-free multi-core later.
- **SIMD**: Not in v0.1. Instance-parallel SIMD for repeated modules in v0.3. See `docs/simd-acceleration.md`.

## Architecture

```
SV Source → slang (parse/elaborate) → Surge IR → LLVM IR → JIT → Native Eval Function
                                                                        ↓
                                                              Simulation Runtime
                                                           (clock loop, state, trace)
```

## Ecosystem (future)

1. **Surge simulator** - this project
2. **Surge testbench** - C++20 coroutine-based testbench API (cocotb-like, native speed)
3. **Python testbench** - direct Python integration
4. **Waveform viewer** - separate project, purpose-built for Surge

## Performance Strategy (ordered by priority)

1. LLVM JIT native code generation (eliminate C++ compile bottleneck)
2. Cycle-based evaluation (no event queue for synchronous logic)
3. Two-state fast path (native 64-bit bitwise ops)
4. Aggressive inlining (flatten hierarchy)
5. SIMD for repeated modules (see `docs/simd-acceleration.md`)
6. Cache-friendly SoA data layout
7. Lock-free multi-threading
8. Arena allocation
9. Incremental compilation

## Sprint Workflow
When done with a sprint: document it briefly in the Sprint Log below, review for any hacks introduced during the sprint and fix them, then create a PR.

## Sprint Log
Once the full pipeline is working, include performance metrics vs verilator.

### Sprint 1: v0.1 Scaffold (complete)
- Full pipeline working: slang parse → Surge IR → LLVM IR → JIT → simulate
- 8-bit counter simulates correctly (20 cycles post-reset, count=20)
- VCD waveform output working
- Combinational adder with concat LHS generates correct LLVM IR
- Build: CMake + FetchContent(slang master) + llvm-config(LLVM 21), Homebrew clang on macOS ARM64
- Key build fix: libc++ from `${LLVM_LIB_DIR}/c++/` (not `lib/` directly) for `__hash_memory` symbol

### Sprint 2: SV Coverage, Optimization, Benchmarking (complete)
- Added: element select (`sig[i]`), range select (`sig[7:0]`), case statements, parameters/localparams, reduction ops, logical operators
- LLVM O2 optimization pass pipeline (PassBuilder), `--no-opt` flag
- Performance timing (cycles/sec in MHz)
- Proper slang diagnostic reporting (TextDiagnosticClient)
- Fixed: LLVM verification failure now aborts, `stoull` error handling, IndexedDown bounds check

**Performance (LFSR 8-bit, 10M cycles, macOS ARM64):**
| Simulator | Throughput | Correctness |
|-----------|-----------|-------------|
| Surge (O2) | **101 MHz** | lfsr_out=0xf4 |
| Verilator 5.045 (-O2) | 39 MHz | lfsr_out=0xf4 |
| **Speedup** | **2.6x** | PASS |

### Sprint 3: Unpacked Arrays, For Loops, ALU+Regfile (complete)
- Added: unpacked array declarations (flattened to N signals), for-loop unrolling (compile-time), dynamic array read (ArrayElement IR + computed GEP), dynamic array write (mux-guarded N assignments)
- VCD multi-character identifiers (base-94 encoding, supports >8000 signals)
- Bug fix: loop variable constant propagation into indexed assignments
- Code quality: signed arithmetic for array range offsets, loop unroll limit warning, loop var type safety

**Performance (10M cycles, macOS ARM64):**
| Design | Surge (O2) | Verilator 5.045 | Speedup | Correctness |
|--------|-----------|-----------------|---------|-------------|
| LFSR 8-bit | **100 MHz** | 39 MHz | **2.6x** | PASS |
| ALU+8x32 Regfile | 18.7 MHz | **33 MHz** | 0.58x | PASS |

ALU regression: mux-guarded dynamic writes are 8x per register write. Needs computed store optimization.

### Sprint 4: Computed Store Optimization (complete)
- Replaced mux-guarded dynamic array writes (N assignments) with single computed GEP store
- Extended `Assignment` struct with optional `indexExpr`, `arraySize`, `elementWidth` for array stores
- Added `emitArrayStore()` codegen method mirroring `ArrayElement` load pattern
- Fixed `lowerConditional()` and `lowerCase()` to partition array-store assignments from scalar merge logic
- Optimized `commitFFs()`: precomputed contiguous FF regions for bulk memcpy (eliminates per-signal branching)

**Performance (1M cycles, macOS ARM64):**
| Design | Surge Sprint 3 | Surge Sprint 4 | Verilator 5.045 | Speedup |
|--------|---------------|----------------|-----------------|---------|
| LFSR 8-bit | 100 MHz | **133 MHz** | 30 MHz | **4.4x** |
| ALU+8x32 Regfile | 18.7 MHz | **75 MHz** | 29 MHz | **2.6x** |

ALU: 4x speedup from computed store. Both designs now faster than Verilator.

### Sprint 5: Self-Stimulus Benchmarks, Pipeline, Hierarchy (complete)
- Added self-stimulus ALU benchmark (`alu_regfile_stim.sv`): embedded 32-bit LFSR generates pseudo-random op/rd/rs1/rs2/imm every cycle
- Added 3-stage pipeline datapath (`pipeline_datapath.sv`): LFSR instruction generator, decode, ALU execute (combinational), writeback with checksum accumulator (28 signals, 12 processes)
- Added multi-module hierarchy support: recursive inline flattening via `lowerInstance` lambda
  - SymbolMap-based signal resolution (replaces string-based `findSignal()` for correctness)
  - Port binding via slang `getPortConnections()` API
  - Handles input ports (NamedValueExpression), output ports (AssignmentExpression), and Conversion wrappers
- Test: `counter_hier.sv` — two `inner_counter` instances, both count correctly

**Performance (1M cycles, macOS ARM64):**
| Design | Surge (O2) | Verilator 5.045 | Speedup | Correctness |
|--------|-----------|-----------------|---------|-------------|
| LFSR 8-bit | **132 MHz** | 40 MHz | **3.3x** | PASS |
| ALU+Regfile (self-stim) | **65 MHz** | 27 MHz | **2.4x** | PASS |
| 3-Stage Pipeline | **33 MHz** | 21 MHz | **1.6x** | PASS |

Surge beats Verilator on all three designs with real LFSR-driven stimulus.

### Sprint 6: SSA Signal Promotion + JIT Simulation Loop (complete)
- **SSA signal promotion**: Pre-load all non-comb signals once at function entry; compute comb signals as pure SSA values in topological order; sequential processes read SSA values instead of re-loading from memory. Eliminates redundant loads (e.g., `lfsr` loaded 10x → 1x, `s2_op` 8x → 1x).
- **Topological sort**: Combinational assignments sorted by signal dependencies (Kahn's algorithm). Detects combinational loops and falls back to original order.
- **JIT simulation loop**: Generated `simulate(state, next_state, cycles)` function with eval body + inline commitFFs (LLVM memcpy intrinsics) in a tight loop. Eliminates per-cycle C++ function call overhead, clock toggle, and runtime commitFFs call. `noalias` attributes on state/next_state pointers enable aggressive LLVM optimization.
- Runtime uses `simulateFn` for no-VCD fast path, falls back to per-cycle `eval()` for VCD tracing.

**Performance (1M cycles, macOS ARM64):**
| Design | Sprint 5 | Sprint 6 | Verilator 5.045 | Speedup | Correctness |
|--------|----------|----------|-----------------|---------|-------------|
| LFSR 8-bit | 132 MHz | **262 MHz** | 39 MHz | **6.7x** | PASS |
| ALU+Regfile (self-stim) | 65 MHz | **126 MHz** | 26 MHz | **4.9x** | PASS |
| 3-Stage Pipeline | 33 MHz | **130 MHz** | 22 MHz | **5.8x** | PASS |

Pipeline: 3.9x improvement from Sprint 5 (33→130 MHz). Speedup now consistent across design complexity.

### Sprint 7: Signed Arithmetic + RISC-V 5-Stage Pipeline (complete)
- **Signed arithmetic**: `isSigned` flag on IR expressions, propagated from slang type system. Codegen emits `ICmpSLT/SLE/SGT/SGE` for signed comparisons, `CreateSExt` for sign-extension, `SDiv/SRem` for signed div/mod.
- **$signed/$unsigned system calls**: `CallExpression` handling in builder — lowers argument with signedness flag.
- **Replication operator**: `{N{expr}}` lowered as Concat with N copies.
- **New binary ops**: `Div`, `Mod` in IR + codegen. `ArithmeticShiftLeft` mapped to `Shl`.
- **RISC-V 5-stage pipeline** (`riscv_pipeline.sv`): LFSR instruction generator, IF/ID/EX/MEM/WB stages, 32x32 register file (x0 hardwired), data forwarding (MEM→ID, WB→ID), full RV32I ALU (ADD/SUB/SLL/SLT/SLTU/XOR/SRL/SRA/OR/AND + LUI), sign-extension via replication, checksum accumulator. 70 signals, 21 processes, 215 bytes state.

**Performance (1M cycles, macOS ARM64):**
| Design | Surge (O2) | Verilator 5.045 | Speedup | Correctness |
|--------|-----------|-----------------|---------|-------------|
| LFSR 8-bit | **268 MHz** | 23 MHz | **11.6x** | PASS |
| ALU+Regfile (self-stim) | **125 MHz** | 33 MHz | **3.8x** | PASS |
| 3-Stage Pipeline | **131 MHz** | 24 MHz | **5.4x** | PASS |
| RISC-V 5-Stage Pipeline | **51 MHz** | 15 MHz | **3.3x** | PASS |

RISC-V pipeline is the most complex design yet — 70 signals including a 32-entry register file with forwarding. Surge maintains 3.3x speedup at this scale.

### Sprint 8: Identity-Mux Elimination (complete)
- **Pre-copy optimization**: In `generateSimulate()`, copy FF regions from state→nextState before eval. This ensures all FFs start with current values, making identity stores redundant.
- **Identity-mux detection**: Detect `Mux(cond, val, SignalRef(target))` patterns in sequential assignments. When the false-branch is a self-reference (keep current value), emit conditional store only for the true-branch. Handles nested identity mux chains.
- **Threshold gating**: Only enable pre-copy when ≥4 identity muxes are detected, avoiding overhead on simple designs.
- Primary beneficiary: RISC-V pipeline with 32x32 register file — 32 identity copies per cycle eliminated.

**Performance (1M cycles, macOS ARM64):**
| Design | Sprint 7 | Sprint 8 | Verilator 5.045 | Speedup | Correctness |
|--------|----------|----------|-----------------|---------|-------------|
| LFSR 8-bit | 268 MHz | **265 MHz** | 26 MHz | **10.2x** | PASS |
| ALU+Regfile (self-stim) | 125 MHz | **130 MHz** | 32 MHz | **4.1x** | PASS |
| 3-Stage Pipeline | 131 MHz | **138 MHz** | 29 MHz | **4.8x** | PASS |
| RISC-V 5-Stage Pipeline | 51 MHz | **68 MHz** | 22 MHz | **3.1x** | PASS |

RISC-V pipeline: 33% improvement from identity-mux elimination (51→68 MHz).

### Sprint 9: SV Feature Coverage (complete)
- **Generate blocks**: Recursive `collectMembers` helper flattens `GenerateBlockSymbol`/`GenerateBlockArraySymbol` scopes. All member iteration (variables, nets, instances, processes) uses flattened list. Port iteration still uses `body.members()` directly (ports aren't inside generate blocks).
- **Constant expression evaluation**: `extractConstantInt()` enhanced with `expr.getConstant()` fallback for slang pre-computed values (genvar arithmetic like `i+1`).
- **Enum types**: `EnumValueSymbol` handling in `lowerNamedValue()` — extracts integer value via `getValue()`.
- **Parameter overrides**: Already work via slang elaboration (no Surge changes needed).
- New tests: `param_adder.sv` (parameterized adder with `#(.WIDTH(16))`), `generate_chain.sv` (generate-for chain of 4 adders), `enum_fsm.sv` (FSM with typedef enum). All verified cycle-accurate against Verilator.

### Sprint 10: Packed Bit/Range Operations + Bugfixes (complete)
- **Packed bit assignment**: `sig[idx] <= val` on packed types. Read-modify-write pattern: `sig = (sig & ~(1<<idx)) | ((val&1)<<idx)`. Supports both constant and dynamic indices.
- **Dynamic range select**: `sig[start+:W]` and `sig[start-:W]` with runtime start. Lowered as `(src >> start) & mask`.
- **Packed range assignment**: `sig[start+:W] <= val` on LHS. Read-modify-write: `sig = (sig & ~(mask<<lo)) | ((val&mask)<<lo)`. Supports Simple, IndexedUp, IndexedDown with constant or dynamic bounds.
- **Bugfix: combinational settle after FF update**: Added final `evalFn_` call after simulation loop, and re-eval in VCD path after commitFFs. Previously, combinational outputs driven from FFs were one cycle stale.
- **Bugfix: unsigned `>>>`**: `ArithmeticShiftRight` on unsigned operands now maps to logical shift right (same as `>>`). Previously emitted `CreateAShr` unconditionally, incorrectly sign-extending unsigned values.
- New test: `barrel_shifter.sv` — 32-bit barrel shifter with SHL/SHR/SRA/rotate, dynamic byte extraction, packed bit/range writes, checksum accumulator. Verified cycle-accurate against Verilator.

**Performance (1M cycles, macOS ARM64):**
| Design | Surge (O2) | Verilator 5.045 | Speedup | Correctness |
|--------|-----------|-----------------|---------|-------------|
| LFSR 8-bit | **268 MHz** | 23 MHz | **11.6x** | PASS |
| ALU+Regfile | **128 MHz** | 33 MHz | **3.9x** | PASS |
| RISC-V 5-Stage | **67 MHz** | 15 MHz | **4.5x** | PASS |
| Barrel Shifter | **138 MHz** | — | — | PASS |

### Sprint 11: System Functions + Dead Signal Elimination (complete)
- **System function expansion**: `$countones` via new `UnaryOp::Popcount` + LLVM `ctpop` intrinsic. `$onehot`/`$onehot0` via popcount comparison. `$isunknown` → constant 0 (2-state sim). Generic `getConstant()` fallback for compile-time system functions (`$clog2`, `$bits`, etc.).
- **Unbased unsized integer literals**: `'0`, `'1` — `UnbasedUnsizedIntegerLiteral` expression handling.
- **Dead signal elimination**: BFS-based liveness analysis from output/input/clock roots. Removes assignments to signals not transitively referenced by any live signal. Called automatically before codegen.
- New test: `fifo.sv` — parameterized synchronous FIFO with `$clog2`, pointer arithmetic, full/empty flags, LFSR self-stimulus, checksum accumulator. 20 signals, 10 processes, 56 bytes. Verified cycle-accurate against Verilator.

**Performance (1M cycles, macOS ARM64):**
| Design | Surge (O2) | Verilator 5.045 | Speedup | Correctness |
|--------|-----------|-----------------|---------|-------------|
| LFSR 8-bit | **268 MHz** | 23 MHz | **11.6x** | PASS |
| ALU+Regfile | **128 MHz** | 33 MHz | **3.9x** | PASS |
| RISC-V 5-Stage | **67 MHz** | 15 MHz | **4.5x** | PASS |
| Barrel Shifter | **138 MHz** | — | — | PASS |
| FIFO (8x32) | **87 MHz** | — | — | PASS |

### Sprint 12: Block Value Tracking + Bugfixes + CRC-32 (complete)
- **Block value tracking for always_comb**: Procedural blocking assignments now propagate values within the same block. Reads after writes in `always_comb` see the updated expression, enabling for-loop accumulation patterns (CRC, scan chains). Save/restore around if/else and case branches for correct conditional semantics.
- **Bugfix: logical NOT on multi-bit signals**: `!x` now correctly lowered as `(x == 0)` instead of bitwise invert `~x`. Previously gave wrong results for any multi-bit operand.
- **Concatenation LHS in procedural blocks**: `{a, b} <= expr` now works in `always_ff`/`always_comb` blocks, not just continuous assigns.
- **Pre/post increment/decrement operators**: `i++`, `++i`, `i--`, `--i` lowered as add/subtract by 1.
- New test: `crc32.sv` — CRC-32 with Ethernet polynomial, 8-bit byte processing per cycle, LFSR self-stimulus, checksum accumulator. Exercises always_comb for-loop accumulation. Verified cycle-accurate against Verilator.

**Performance (1M cycles, macOS ARM64):**
| Design | Surge (O2) | Verilator 5.045 | Speedup | Correctness |
|--------|-----------|-----------------|---------|-------------|
| LFSR 8-bit | **262 MHz** | 23 MHz | **11.4x** | PASS |
| ALU+Regfile | **126 MHz** | 33 MHz | **3.8x** | PASS |
| RISC-V 5-Stage | **67 MHz** | 15 MHz | **4.5x** | PASS |
| Barrel Shifter | **139 MHz** | — | — | PASS |
| FIFO (8x32) | **88 MHz** | — | — | PASS |
| CRC-32 | **58 MHz** | — | — | PASS |

## Research

Full state-of-the-art research is in `docs/simd-acceleration.md` and was captured during initial planning. Key references:
- **Verilator**: Cycle-based compiled sim, fastest open-source. Our baseline to beat.
- **slang**: Most complete open-source SV parser. Our frontend.
- **GHDL/NVC**: Prove LLVM JIT works for HDL simulation.
- **VCS**: Hybrid cycle-based/event-driven, cross-module inlining, event coalescing.
- **Xcelium**: Lock-free multi-core, epoch-based sync, design partitioning.
- **RepCut** (ASPLOS 2023): Replication-aided partitioning for cache-friendly parallel sim.
- **Manticore** (ASPLOS 2024): Static BSP parallel sim with LLVM.
- **ESSENT**: FIRRTL → LLVM compilation for simulation.
