#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SURGE="$PROJECT_DIR/build/surge"
LFSR_SV="$PROJECT_DIR/tests/lfsr.sv"
ALU_SV="$PROJECT_DIR/tests/alu_regfile.sv"

CYCLES=${1:-1000000}

echo "=========================================="
echo " Surge vs Verilator Benchmark"
echo "=========================================="
echo ""

# ── LFSR Benchmark ─────────────────────────────────────────────────────────

echo "=== Design: LFSR (8-bit Galois) ==="
echo "Cycles: $CYCLES"
echo ""

echo "--- Surge (O2) ---"
SURGE_LFSR=$($SURGE "$LFSR_SV" --cycles "$CYCLES" 2>&1)
echo "$SURGE_LFSR"
echo ""

# ── ALU + Register File Benchmark ──────────────────────────────────────────

echo "=== Design: ALU + 8x32-bit Register File ==="
echo "Cycles: $CYCLES"
echo ""

echo "--- Surge (O2) ---"
SURGE_ALU=$($SURGE "$ALU_SV" --cycles "$CYCLES" 2>&1)
echo "$SURGE_ALU"
echo ""

# ── Verilator ──────────────────────────────────────────────────────────────

if command -v verilator &>/dev/null; then
    VERI_DIR=$(mktemp -d)

    # ── Verilator LFSR ─────────────────────────────────────────────────────

    echo "--- Verilator: LFSR ---"

    cat > "$VERI_DIR/tb_lfsr.cpp" << 'CPPEOF'
#include "Vlfsr.h"
#include "verilated.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vlfsr* dut = new Vlfsr;

    uint64_t cycles = 1000000;
    if (argc > 1) cycles = strtoull(argv[1], nullptr, 10);

    // Reset
    dut->rst = 1;
    for (int i = 0; i < 5; i++) {
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }
    dut->rst = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < cycles; i++) {
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    printf("  lfsr_out = %u (0x%02x)\n", dut->lfsr_out, dut->lfsr_out);
    printf("  Cycles: %llu\n", (unsigned long long)cycles);
    printf("  Wall time: %.6f s\n", elapsed.count());
    printf("  Throughput: %.2f MHz\n", cycles / elapsed.count() / 1e6);

    delete dut;
    return 0;
}
CPPEOF

    verilator --cc "$LFSR_SV" \
        --exe "$VERI_DIR/tb_lfsr.cpp" \
        -Mdir "$VERI_DIR/obj_dir_lfsr" \
        --build -j 4 \
        -CFLAGS "-O2" 2>/dev/null

    VERI_LFSR=$("$VERI_DIR/obj_dir_lfsr/Vlfsr" "$CYCLES")
    echo "$VERI_LFSR"
    echo ""

    # ── Verilator ALU ──────────────────────────────────────────────────────

    echo "--- Verilator: ALU + Regfile ---"

    cat > "$VERI_DIR/tb_alu.cpp" << 'CPPEOF'
#include "Valu_regfile.h"
#include "verilated.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Valu_regfile* dut = new Valu_regfile;

    uint64_t cycles = 1000000;
    if (argc > 1) cycles = strtoull(argv[1], nullptr, 10);

    // Reset
    dut->rst = 1;
    dut->op = 0; dut->rd = 0; dut->rs1 = 0; dut->rs2 = 0; dut->imm = 0;
    for (int i = 0; i < 5; i++) {
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }
    dut->rst = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < cycles; i++) {
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    printf("  result = %u (0x%08x)\n", dut->result, dut->result);
    printf("  Cycles: %llu\n", (unsigned long long)cycles);
    printf("  Wall time: %.6f s\n", elapsed.count());
    printf("  Throughput: %.2f MHz\n", cycles / elapsed.count() / 1e6);

    delete dut;
    return 0;
}
CPPEOF

    verilator --cc "$ALU_SV" \
        --exe "$VERI_DIR/tb_alu.cpp" \
        -Mdir "$VERI_DIR/obj_dir_alu" \
        --build -j 4 \
        -CFLAGS "-O2" 2>/dev/null

    VERI_ALU=$("$VERI_DIR/obj_dir_alu/Valu_regfile" "$CYCLES")
    echo "$VERI_ALU"
    echo ""

    # Cleanup
    rm -rf "$VERI_DIR"

    # ── Correctness Check ──────────────────────────────────────────────────

    echo "=== Correctness Check ==="
    SURGE_LFSR_VAL=$(echo "$SURGE_LFSR" | grep "lfsr_out" | head -1 | grep -o '0x[0-9a-f]*')
    VERI_LFSR_VAL=$(echo "$VERI_LFSR" | grep "lfsr_out" | head -1 | grep -o '0x[0-9a-f]*')
    echo "  LFSR: Surge=$SURGE_LFSR_VAL Verilator=$VERI_LFSR_VAL"
    if [ "$SURGE_LFSR_VAL" = "$VERI_LFSR_VAL" ]; then
        echo "  LFSR: PASS"
    else
        echo "  LFSR: FAIL"
    fi

    SURGE_ALU_VAL=$(echo "$SURGE_ALU" | grep "  result = " | head -1 | awk '{print $3}')
    VERI_ALU_VAL=$(echo "$VERI_ALU" | grep "result = " | head -1 | awk '{print $3}')
    echo "  ALU:  Surge=$SURGE_ALU_VAL Verilator=$VERI_ALU_VAL"
    if [ "$SURGE_ALU_VAL" = "$VERI_ALU_VAL" ]; then
        echo "  ALU:  PASS"
    else
        echo "  ALU:  FAIL"
    fi
else
    echo "Verilator not found -- skipping Verilator benchmark."
    echo "Install: brew install verilator"
fi
