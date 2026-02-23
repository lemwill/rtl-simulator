#pragma once
#include "ir.h"
#include <string>
#include <memory>
#include <vector>

namespace surge::ir {

/// Parse a SystemVerilog file with slang and lower the top-level module to Surge IR.
std::unique_ptr<Module> buildFromFile(const std::string& path);

/// Parse multiple SystemVerilog files and lower the top-level module to Surge IR.
std::unique_ptr<Module> buildFromFiles(const std::vector<std::string>& paths);

} // namespace surge::ir
