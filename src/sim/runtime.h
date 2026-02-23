#pragma once
#include "../ir/ir.h"
#include "../codegen/codegen.h"
#include "../trace/vcd_writer.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace surge::sim {

struct SimConfig {
    uint64_t maxCycles = 100;
    std::string vcdPath;    // empty = no trace
    bool dumpIR = false;
};

struct SimResult {
    uint64_t cyclesSimulated = 0;
    double wallTimeSeconds = 0.0;
    double cyclesPerSecond = 0.0;
};

class Runtime {
public:
    Runtime(const ir::Module& mod, codegen::EvalFn evalFn, const SimConfig& cfg);
    ~Runtime();

    /// Run the simulation for cfg.maxCycles clock cycles.
    SimResult run();

    /// Read a signal value from current state.
    uint64_t readSignal(const std::string& name) const;
    uint64_t readSignal(uint32_t index) const;

    /// Write a signal value to current state (for driving inputs).
    void writeSignal(const std::string& name, uint64_t val);
    void writeSignal(uint32_t index, uint64_t val);

private:
    const ir::Module& mod_;
    codegen::EvalFn evalFn_;
    SimConfig cfg_;

    std::vector<uint8_t> state_;
    std::vector<uint8_t> nextState_;
    std::unique_ptr<trace::VCDWriter> vcd_;
    uint64_t time_ = 0;

    void commitFFs();
    void traceAll();
};

} // namespace surge::sim
