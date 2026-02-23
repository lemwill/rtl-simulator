#pragma once
#include "../ir/ir.h"
#include <cstdint>
#include <memory>
#include <string>

namespace surge::codegen {

/// Function pointer type: eval(state*, next_state*) — single cycle
using EvalFn = void (*)(uint8_t*, uint8_t*);

/// Function pointer type: simulate(state*, next_state*, cycles) — JIT loop
using SimulateFn = void (*)(uint8_t*, uint8_t*, uint64_t);

/// Compiled module: holds JIT state and the eval function pointer.
class CompiledModule {
public:
    ~CompiledModule();
    CompiledModule(CompiledModule&&) noexcept;
    CompiledModule& operator=(CompiledModule&&) noexcept;

    EvalFn evalFn() const { return evalFn_; }
    SimulateFn simulateFn() const { return simulateFn_; }
    const std::string& irDump() const { return irDump_; }

    // Non-copyable
    CompiledModule(const CompiledModule&) = delete;
    CompiledModule& operator=(const CompiledModule&) = delete;

private:
    friend class Compiler;
    CompiledModule() = default;

    EvalFn evalFn_ = nullptr;
    SimulateFn simulateFn_ = nullptr;
    std::string irDump_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Compile a Surge IR module to native code via LLVM JIT.
class Compiler {
public:
    Compiler();
    ~Compiler();

    /// Compile module IR → LLVM IR → native. Returns compiled module with eval fn.
    CompiledModule compile(const ir::Module& mod);

    /// If true, dump LLVM IR to stderr before JIT.
    void setDumpIR(bool v) { dumpIR_ = v; }

    /// Set the LLVM optimization level (0 = none, 2 = O2). Default is 2.
    void setOptLevel(int level) { optLevel_ = level; }

private:
    bool dumpIR_ = false;
    int optLevel_ = 2;
};

} // namespace surge::codegen
