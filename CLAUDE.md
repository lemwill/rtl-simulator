# Surge - The Fastest Open-Source RTL Simulator

## What To Do Next

**Sprint 5 complete.** Self-stimulus benchmarks, 3-stage pipeline, multi-module hierarchy.

Next priorities:
1. **Generate blocks**: Parameterized module generation.
2. **Wider benchmark designs**: More complex designs (pipelined CPU, bus arbiter) to stress-test at scale.
3. **Combinational loop detection**: Prevent infinite loops in pure-combinational designs.
4. **Signed arithmetic**: `signed` type handling and arithmetic right shift.

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
