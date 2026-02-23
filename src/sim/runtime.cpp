#include "runtime.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

namespace surge::sim {

Runtime::Runtime(const ir::Module& mod, codegen::EvalFn evalFn, const SimConfig& cfg)
    : mod_(mod), evalFn_(evalFn), cfg_(cfg)
{
    state_.resize(mod.stateSize, 0);
    nextState_.resize(mod.stateSize, 0);
    buildFFRegions();

    if (!cfg_.vcdPath.empty()) {
        vcd_ = std::make_unique<trace::VCDWriter>(cfg_.vcdPath);
        vcd_->writeHeader(mod);
    }
}

Runtime::~Runtime() = default;

SimResult Runtime::run() {
    SimResult result;
    auto* clkSig = mod_.findSignal("clk");
    auto* rstSig = mod_.findSignal("rst");

    if (!clkSig) {
        std::cerr << "surge: no 'clk' signal found, cannot run cycle-based sim\n";
        return result;
    }

    // Reset phase: hold rst=1 for 5 cycles (not timed)
    if (rstSig) {
        writeSignal(rstSig->index, 1);
        for (int i = 0; i < 5; i++) {
            writeSignal(clkSig->index, 1);
            evalFn_(state_.data(), nextState_.data());
            commitFFs();
            traceAll();
            time_ += 5;

            writeSignal(clkSig->index, 0);
            traceAll();
            time_ += 5;
        }
        writeSignal(rstSig->index, 0);
    }

    // Main simulation loop (timed)
    auto startTime = std::chrono::high_resolution_clock::now();

    for (uint64_t cycle = 0; cycle < cfg_.maxCycles; cycle++) {
        writeSignal(clkSig->index, 1);
        evalFn_(state_.data(), nextState_.data());
        commitFFs();
        traceAll();
        time_ += 5;

        writeSignal(clkSig->index, 0);
        traceAll();
        time_ += 5;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    result.cyclesSimulated = cfg_.maxCycles;
    result.wallTimeSeconds = elapsed.count();
    result.cyclesPerSecond = (elapsed.count() > 0)
        ? static_cast<double>(cfg_.maxCycles) / elapsed.count()
        : 0.0;

    std::cerr << "surge: simulation complete (" << cfg_.maxCycles << " cycles in "
              << result.wallTimeSeconds << "s, "
              << result.cyclesPerSecond / 1e6 << " MHz)\n";

    return result;
}

uint64_t Runtime::readSignal(const std::string& name) const {
    auto* sig = mod_.findSignal(name);
    if (!sig) return 0;
    return readSignal(sig->index);
}

uint64_t Runtime::readSignal(uint32_t index) const {
    auto& sig = mod_.signals[index];
    uint32_t bytes = ir::bytesForWidth(sig.width);
    uint64_t val = 0;
    std::memcpy(&val, state_.data() + sig.stateOffset, bytes);
    // Mask to actual width
    if (sig.width < 64)
        val &= (1ULL << sig.width) - 1;
    return val;
}

void Runtime::writeSignal(const std::string& name, uint64_t val) {
    auto* sig = mod_.findSignal(name);
    if (!sig) return;
    writeSignal(sig->index, val);
}

void Runtime::writeSignal(uint32_t index, uint64_t val) {
    auto& sig = mod_.signals[index];
    uint32_t bytes = ir::bytesForWidth(sig.width);
    if (sig.width < 64)
        val &= (1ULL << sig.width) - 1;
    std::memcpy(state_.data() + sig.stateOffset, &val, bytes);
}

void Runtime::buildFFRegions() {
    // Collect (offset, bytes) for each FF signal
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for (auto& sig : mod_.signals) {
        if (!sig.isFF) continue;
        ranges.push_back({sig.stateOffset, ir::bytesForWidth(sig.width)});
    }
    // Sort by offset
    std::sort(ranges.begin(), ranges.end());
    // Merge contiguous ranges into bulk regions
    for (auto& [off, len] : ranges) {
        if (!ffRegions_.empty() && ffRegions_.back().offset + ffRegions_.back().bytes == off) {
            ffRegions_.back().bytes += len;
        } else {
            ffRegions_.push_back({off, len});
        }
    }
}

void Runtime::commitFFs() {
    for (auto& r : ffRegions_) {
        std::memcpy(state_.data() + r.offset,
                    nextState_.data() + r.offset, r.bytes);
    }
}

void Runtime::traceAll() {
    if (!vcd_) return;
    vcd_->writeTimestep(time_);
    for (auto& sig : mod_.signals) {
        uint64_t val = readSignal(sig.index);
        vcd_->writeSignal(sig, val);
    }
}

} // namespace surge::sim
