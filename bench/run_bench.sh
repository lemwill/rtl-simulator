#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SURGE="$PROJECT_DIR/build/surge"

CYCLES=${1:-1000000}

echo "=========================================="
echo " Surge vs Verilator Benchmark"
echo "=========================================="
echo ""

# ── Design list ───────────────────────────────────────────────────────────────

declare -a DESIGNS=("lfsr" "alu_regfile_stim" "pipeline_datapath")
declare -a LABELS=("LFSR (8-bit Galois)" "ALU+Regfile (self-stim)" "3-Stage Pipeline")
declare -a SV_FILES=("$PROJECT_DIR/tests/lfsr.sv" "$PROJECT_DIR/tests/alu_regfile_stim.sv" "$PROJECT_DIR/tests/pipeline_datapath.sv")
declare -a CHECK_SIGNALS=("lfsr_out" "result" "checksum")

# ── Surge Benchmarks ─────────────────────────────────────────────────────────

declare -a SURGE_RESULTS
for i in "${!DESIGNS[@]}"; do
    echo "=== Design: ${LABELS[$i]} ==="
    echo "Cycles: $CYCLES"
    echo ""
    echo "--- Surge (O2) ---"
    SURGE_RESULTS[$i]=$($SURGE "${SV_FILES[$i]}" --cycles "$CYCLES" 2>&1)
    echo "${SURGE_RESULTS[$i]}"
    echo ""
done

# ── Verilator Benchmarks ─────────────────────────────────────────────────────

if command -v verilator &>/dev/null; then
    VERI_DIR=$(mktemp -d)
    declare -a VERI_RESULTS

    # Helper: generate a simple clk/rst testbench for any module
    gen_tb() {
        local MODULE_NAME=$1
        local HEADER_NAME=$2
        local CHECK_SIG=$3
        local TB_FILE=$4
        cat > "$TB_FILE" << CPPEOF
#include "${HEADER_NAME}.h"
#include "verilated.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    ${HEADER_NAME}* dut = new ${HEADER_NAME};

    uint64_t cycles = 1000000;
    if (argc > 1) cycles = strtoull(argv[1], nullptr, 10);

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

    printf("  ${CHECK_SIG} = %u (0x%08x)\n", dut->${CHECK_SIG}, dut->${CHECK_SIG});
    printf("  Cycles: %llu\n", (unsigned long long)cycles);
    printf("  Wall time: %.6f s\n", elapsed.count());
    printf("  Throughput: %.2f MHz\n", cycles / elapsed.count() / 1e6);

    delete dut;
    return 0;
}
CPPEOF
    }

    for i in "${!DESIGNS[@]}"; do
        MODULE_NAME="${DESIGNS[$i]}"
        HEADER_NAME="V${MODULE_NAME}"
        CHECK_SIG="${CHECK_SIGNALS[$i]}"
        TB_FILE="$VERI_DIR/tb_${MODULE_NAME}.cpp"
        OBJ_DIR="$VERI_DIR/obj_dir_${MODULE_NAME}"

        echo "--- Verilator: ${LABELS[$i]} ---"

        gen_tb "$MODULE_NAME" "$HEADER_NAME" "$CHECK_SIG" "$TB_FILE"

        verilator --cc "${SV_FILES[$i]}" \
            --exe "$TB_FILE" \
            -Mdir "$OBJ_DIR" \
            --build -j 4 \
            -CFLAGS "-O2" 2>/dev/null

        VERI_RESULTS[$i]=$("$OBJ_DIR/$HEADER_NAME" "$CYCLES")
        echo "${VERI_RESULTS[$i]}"
        echo ""
    done

    # Cleanup
    rm -rf "$VERI_DIR"

    # ── Correctness Check ─────────────────────────────────────────────────────

    echo "=== Correctness Check ==="
    ALL_PASS=true
    for i in "${!DESIGNS[@]}"; do
        SIG="${CHECK_SIGNALS[$i]}"
        SURGE_VAL=$(echo "${SURGE_RESULTS[$i]}" | grep "  ${SIG} = " | head -1 | awk '{print $3}')
        VERI_VAL=$(echo "${VERI_RESULTS[$i]}" | grep "${SIG} = " | head -1 | awk '{print $3}')
        echo "  ${LABELS[$i]}: Surge=$SURGE_VAL Verilator=$VERI_VAL"
        if [ "$SURGE_VAL" = "$VERI_VAL" ]; then
            echo "  ${LABELS[$i]}: PASS"
        else
            echo "  ${LABELS[$i]}: FAIL"
            ALL_PASS=false
        fi
    done

    if $ALL_PASS; then
        echo ""
        echo "All correctness checks PASSED."
    fi
else
    echo "Verilator not found -- skipping Verilator benchmark."
    echo "Install: brew install verilator"
fi
