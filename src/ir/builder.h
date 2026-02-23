#pragma once
#include "ir.h"
#include <string>
#include <memory>

namespace surge::ir {

/// Parse a SystemVerilog file with slang and lower the top-level module to Surge IR.
std::unique_ptr<Module> buildFromFile(const std::string& path);

} // namespace surge::ir
