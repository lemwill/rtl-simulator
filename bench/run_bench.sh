#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SURGE="$PROJECT_DIR/build/surge"
LFSR_SV="$PROJECT_DIR/tests/lfsr.sv"

CYCLES=${1:-1000000}

echo "=== Surge vs Verilator Benchmark ==="
echo "Design: LFSR (8-bit Galois)"
echo "Cycles: $CYCLES"
echo ""

# ── Surge (O2) ──────────────────────────────────────────────────────────────

echo "--- Surge (O2) ---"
SURGE_OUT=$($SURGE "$LFSR_SV" --cycles "$CYCLES" 2>&1)
echo "$SURGE_OUT"
echo ""

# ── Surge (no-opt) ──────────────────────────────────────────────────────────

echo "--- Surge (no-opt) ---"
SURGE_NOOPT_OUT=$($SURGE "$LFSR_SV" --cycles "$CYCLES" --no-opt 2>&1)
echo "$SURGE_NOOPT_OUT"
echo ""

# ── Verilator ───────────────────────────────────────────────────────────────

if command -v verilator &>/dev/null; then
    echo "--- Verilator ---"

    VERI_DIR=$(mktemp -d)

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

    // Timed simulation
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

    # Build with Verilator
    verilator --cc "$LFSR_SV" \
        --exe "$VERI_DIR/tb_lfsr.cpp" \
        -Mdir "$VERI_DIR/obj_dir" \
        --build -j 4 \
        -CFLAGS "-O2" 2>/dev/null

    # Run
    VERI_OUT=$("$VERI_DIR/obj_dir/Vlfsr" "$CYCLES")
    echo "$VERI_OUT"

    # Cleanup
    rm -rf "$VERI_DIR"
    echo ""

    # ── Correctness Check ───────────────────────────────────────────────────

    SURGE_VAL=$(echo "$SURGE_OUT" | grep "lfsr_out" | head -1 | grep -o '0x[0-9a-f]*')
    VERI_VAL=$(echo "$VERI_OUT" | grep "lfsr_out" | head -1 | grep -o '0x[0-9a-f]*')

    echo "=== Correctness Check ==="
    echo "  Surge lfsr_out:     $SURGE_VAL"
    echo "  Verilator lfsr_out: $VERI_VAL"
    if [ "$SURGE_VAL" = "$VERI_VAL" ]; then
        echo "  PASS: Values match!"
    else
        echo "  FAIL: Values differ!"
        exit 1
    fi
else
    echo "Verilator not found -- skipping Verilator benchmark."
    echo "Install: brew install verilator"
fi
