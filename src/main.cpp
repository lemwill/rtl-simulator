#include "ir/builder.h"
#include "codegen/codegen.h"
#include "sim/runtime.h"

#include <iostream>
#include <string>
#include <vector>

static void usage(const char* prog) {
    std::cerr << "Surge RTL Simulator v0.1\n\n"
              << "Usage: " << prog << " <file.sv> [file2.sv ...] [options]\n\n"
              << "Options:\n"
              << "  --cycles N    Number of simulation cycles (default: 100)\n"
              << "  --vcd FILE    Write VCD waveform to FILE\n"
              << "  --dump-ir     Dump LLVM IR to stderr\n"
              << "  --no-opt      Disable LLVM optimization passes\n"
              << "  --help        Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::vector<std::string> svFiles;
    surge::sim::SimConfig cfg;
    bool noOpt = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--cycles" && i + 1 < argc) {
            try {
                cfg.maxCycles = std::stoull(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "surge: invalid cycle count '" << argv[i] << "'\n";
                return 1;
            }
        } else if (arg == "--vcd" && i + 1 < argc) {
            cfg.vcdPath = argv[++i];
        } else if (arg == "--dump-ir") {
            cfg.dumpIR = true;
        } else if (arg == "--no-opt") {
            noOpt = true;
        } else if (arg[0] != '-') {
            svFiles.push_back(arg);
        } else {
            std::cerr << "surge: unknown option '" << arg << "'\n";
            return 1;
        }
    }

    if (svFiles.empty()) {
        std::cerr << "surge: no input file\n";
        return 1;
    }

    // ── Pipeline: Parse → IR → Codegen → Simulate ──

    // 1. Parse SV and build IR
    auto mod = surge::ir::buildFromFiles(svFiles);
    if (!mod) return 1;

    // 1b. Dead signal elimination
    uint32_t deadRemoved = mod->eliminateDeadSignals();
    if (deadRemoved > 0)
        std::cerr << "surge: eliminated " << deadRemoved << " dead assignments\n";

    // 2. Compile IR → LLVM → native
    surge::codegen::Compiler compiler;
    compiler.setDumpIR(cfg.dumpIR);
    if (noOpt) compiler.setOptLevel(0);
    auto compiled = compiler.compile(*mod);
    if (!compiled.evalFn()) {
        std::cerr << "surge: compilation failed\n";
        return 1;
    }

    // 3. Simulate
    surge::sim::Runtime runtime(*mod, compiled.evalFn(), compiled.simulateFn(), cfg);
    auto simResult = runtime.run();

    // 4. Print final signal values
    std::cout << "\n=== Final State ===\n";
    for (auto& sig : mod->signals) {
        uint64_t val = runtime.readSignal(sig.index);
        std::cout << "  " << sig.name << " = " << val;
        if (sig.width > 1)
            std::cout << " (0x" << std::hex << val << std::dec << ")";
        std::cout << "\n";
    }

    // 5. Print performance
    if (simResult.cyclesSimulated > 0) {
        std::cout << "\n=== Performance ===\n";
        std::cout << "  Cycles: " << simResult.cyclesSimulated << "\n";
        std::cout << "  Wall time: " << simResult.wallTimeSeconds << " s\n";
        std::cout << "  Throughput: " << simResult.cyclesPerSecond / 1e6 << " MHz\n";
    }

    return 0;
}
