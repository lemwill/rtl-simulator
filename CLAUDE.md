# Surge - The Fastest Open-Source RTL Simulator

## What To Do Next

**Sprint 2 complete.** 2.6x faster than Verilator on LFSR benchmark with correct results.

Next priorities:
1. Expand SV coverage: for loops, memories/arrays, multi-module hierarchy, wider bitwidths
2. More complex benchmark designs (FSM with driven inputs, pipelined datapath)
3. Testbench input driving (currently only clock/reset are driven automatically)

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
