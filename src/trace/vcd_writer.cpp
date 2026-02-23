#include "vcd_writer.h"
#include <bitset>

namespace surge::trace {

VCDWriter::VCDWriter(const std::string& path) : file_(path) {
    if (!file_.is_open())
        throw std::runtime_error("surge: cannot open VCD file: " + path);
}

VCDWriter::~VCDWriter() {
    if (file_.is_open())
        file_.close();
}

std::string VCDWriter::idString(uint32_t index) const {
    // Base-94 encoding using printable ASCII ('!' through '~')
    std::string id;
    do {
        id += static_cast<char>('!' + (index % 94));
        index /= 94;
    } while (index > 0);
    return id;
}

void VCDWriter::writeHeader(const ir::Module& mod) {
    file_ << "$date\n  Surge RTL Simulator\n$end\n";
    file_ << "$version\n  Surge v0.1\n$end\n";
    file_ << "$timescale 1ns $end\n";
    file_ << "$scope module " << mod.name << " $end\n";

    for (auto& sig : mod.signals) {
        file_ << "$var ";
        if (sig.kind == ir::SignalKind::Input)
            file_ << "wire";
        else
            file_ << "reg";
        file_ << " " << sig.width << " " << idString(sig.index) << " "
              << sig.name << " $end\n";
    }

    file_ << "$upscope $end\n";
    file_ << "$enddefinitions $end\n";
    file_ << "$dumpvars\n";

    // Initial values (all zero)
    for (auto& sig : mod.signals) {
        if (sig.width == 1) {
            file_ << "0" << idString(sig.index) << "\n";
        } else {
            file_ << "b";
            for (int i = sig.width - 1; i >= 0; i--)
                file_ << "0";
            file_ << " " << idString(sig.index) << "\n";
        }
    }
    file_ << "$end\n";
    headerWritten_ = true;
}

void VCDWriter::writeTimestep(uint64_t time) {
    if (time != lastTime_) {
        file_ << "#" << time << "\n";
        lastTime_ = time;
    }
}

void VCDWriter::writeSignal(const ir::Signal& sig, uint64_t value) {
    if (sig.width == 1) {
        file_ << (value & 1) << idString(sig.index) << "\n";
    } else {
        file_ << "b";
        for (int i = sig.width - 1; i >= 0; i--)
            file_ << ((value >> i) & 1);
        file_ << " " << idString(sig.index) << "\n";
    }
}

} // namespace surge::trace
