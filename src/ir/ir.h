#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace surge::ir {

// ── Signal ──────────────────────────────────────────────────────────────────

enum class SignalKind { Input, Output, Internal };

struct Signal {
    std::string name;
    uint32_t width;       // bit width
    SignalKind kind;
    uint32_t index;       // unique index in module's signal vector
    uint32_t stateOffset; // byte offset into flat state array (set by computeLayout)
    bool isFF = false;    // true if driven by always_ff (sequential element)
};

// ── Expressions ─────────────────────────────────────────────────────────────

enum class ExprKind {
    Constant,
    SignalRef,
    UnaryOp,
    BinaryOp,
    Mux,          // ternary select: cond ? t : f
    Slice,        // bit extract [hi:lo]
    Concat,
    ArrayElement, // dynamic unpacked array element access
};

enum class UnaryOp { Not, Negate, ReduceAnd, ReduceOr, ReduceXor, Popcount, SignExtend };

enum class BinaryOp {
    And, Or, Xor,
    Add, Sub, Mul, Div, Mod,
    Eq, Neq, Lt, Lte, Gt, Gte,
    Shl, Shr, AShr,
};

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Expr {
    ExprKind kind;
    uint32_t width; // result width in bits

    uint64_t constVal       = 0;
    uint32_t signalIndex    = 0;
    UnaryOp  unaryOp        = UnaryOp::Not;
    BinaryOp binaryOp       = BinaryOp::Add;
    uint32_t sliceHi        = 0;
    uint32_t sliceLo        = 0;
    uint32_t arrayBaseIndex = 0; // ArrayElement: signal index of element [0]
    uint32_t arraySize      = 0; // ArrayElement: number of elements
    uint32_t elementWidth   = 0; // ArrayElement: bit width per element
    bool     isSigned       = false; // true for signed comparison/arithmetic
    std::vector<ExprPtr> operands;

    // ── factories ──
    static ExprPtr constant(uint32_t w, uint64_t v);
    static ExprPtr signalRef(uint32_t w, uint32_t idx);
    static ExprPtr unary(UnaryOp op, uint32_t w, ExprPtr a);
    static ExprPtr binary(BinaryOp op, uint32_t w, ExprPtr l, ExprPtr r);
    static ExprPtr mux(uint32_t w, ExprPtr cond, ExprPtr t, ExprPtr f);
    static ExprPtr slice(uint32_t w, ExprPtr src, uint32_t hi, uint32_t lo);
    static ExprPtr concat(uint32_t w, std::vector<ExprPtr> parts);
    static ExprPtr arrayElement(uint32_t elemWidth, uint32_t baseIdx, uint32_t size, ExprPtr indexExpr);
};

// ── Processes ───────────────────────────────────────────────────────────────

struct Assignment {
    uint32_t targetIndex; // signal index (scalar) or base signal index (array store)
    ExprPtr  value;

    // Optional: computed array store (non-null indexExpr => array store)
    ExprPtr  indexExpr;        // runtime index expression
    uint32_t arraySize     = 0; // element count for bounds clamping
    uint32_t elementWidth  = 0; // bit width per element
};

enum class EdgeKind { Posedge, Negedge };

struct Process {
    enum Kind { Combinational, Sequential };
    Kind kind;

    // Sequential only:
    uint32_t clockSignalIndex = 0;
    EdgeKind clockEdge        = EdgeKind::Posedge;

    std::vector<Assignment> assignments;
};

// ── Module ──────────────────────────────────────────────────────────────────

struct Module {
    std::string name;
    std::vector<Signal> signals;
    std::vector<Process> processes;

    // Initial values: signal index → constant value (from initial blocks)
    std::vector<std::pair<uint32_t, uint64_t>> initialValues;

    uint32_t stateSize = 0; // total bytes, set by computeLayout()

    uint32_t addSignal(const std::string& n, uint32_t w, SignalKind k);
    const Signal* findSignal(const std::string& n) const;
    void computeLayout();
    uint32_t eliminateDeadSignals(); // returns number of dead assignments removed
};

// ── Convenience ─────────────────────────────────────────────────────────────

inline uint32_t bytesForWidth(uint32_t bits) {
    if (bits <= 8)  return 1;
    if (bits <= 16) return 2;
    if (bits <= 32) return 4;
    return 8; // up to 64-bit for v0.1
}

} // namespace surge::ir
