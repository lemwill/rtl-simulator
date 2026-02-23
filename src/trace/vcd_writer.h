#pragma once
#include "../ir/ir.h"
#include <cstdint>
#include <fstream>
#include <string>

namespace surge::trace {

class VCDWriter {
public:
    explicit VCDWriter(const std::string& path);
    ~VCDWriter();

    void writeHeader(const ir::Module& mod);
    void writeTimestep(uint64_t time);
    void writeSignal(const ir::Signal& sig, uint64_t value);

private:
    std::ofstream file_;
    bool headerWritten_ = false;
    uint64_t lastTime_ = ~0ULL;

    char idChar(uint32_t index) const;
};

} // namespace surge::trace
