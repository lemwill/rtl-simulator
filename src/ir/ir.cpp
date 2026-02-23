#include "ir.h"

namespace surge::ir {

// ── Expr factories ──────────────────────────────────────────────────────────

ExprPtr Expr::constant(uint32_t w, uint64_t v) {
    auto e = std::make_shared<Expr>();
    e->kind     = ExprKind::Constant;
    e->width    = w;
    e->constVal = v;
    return e;
}

ExprPtr Expr::signalRef(uint32_t w, uint32_t idx) {
    auto e = std::make_shared<Expr>();
    e->kind        = ExprKind::SignalRef;
    e->width       = w;
    e->signalIndex = idx;
    return e;
}

ExprPtr Expr::unary(UnaryOp op, uint32_t w, ExprPtr a) {
    auto e = std::make_shared<Expr>();
    e->kind    = ExprKind::UnaryOp;
    e->width   = w;
    e->unaryOp = op;
    e->operands.push_back(std::move(a));
    return e;
}

ExprPtr Expr::binary(BinaryOp op, uint32_t w, ExprPtr l, ExprPtr r) {
    auto e = std::make_shared<Expr>();
    e->kind     = ExprKind::BinaryOp;
    e->width    = w;
    e->binaryOp = op;
    e->operands.push_back(std::move(l));
    e->operands.push_back(std::move(r));
    return e;
}

ExprPtr Expr::mux(uint32_t w, ExprPtr cond, ExprPtr t, ExprPtr f) {
    auto e = std::make_shared<Expr>();
    e->kind  = ExprKind::Mux;
    e->width = w;
    e->operands.push_back(std::move(cond));
    e->operands.push_back(std::move(t));
    e->operands.push_back(std::move(f));
    return e;
}

ExprPtr Expr::slice(uint32_t w, ExprPtr src, uint32_t hi, uint32_t lo) {
    auto e = std::make_shared<Expr>();
    e->kind    = ExprKind::Slice;
    e->width   = w;
    e->sliceHi = hi;
    e->sliceLo = lo;
    e->operands.push_back(std::move(src));
    return e;
}

ExprPtr Expr::concat(uint32_t w, std::vector<ExprPtr> parts) {
    auto e = std::make_shared<Expr>();
    e->kind     = ExprKind::Concat;
    e->width    = w;
    e->operands = std::move(parts);
    return e;
}

ExprPtr Expr::arrayElement(uint32_t elemWidth, uint32_t baseIdx, uint32_t size, ExprPtr indexExpr) {
    auto e = std::make_shared<Expr>();
    e->kind           = ExprKind::ArrayElement;
    e->width          = elemWidth;
    e->arrayBaseIndex = baseIdx;
    e->arraySize      = size;
    e->elementWidth   = elemWidth;
    e->operands.push_back(std::move(indexExpr));
    return e;
}

// ── Module ──────────────────────────────────────────────────────────────────

uint32_t Module::addSignal(const std::string& n, uint32_t w, SignalKind k) {
    uint32_t idx = static_cast<uint32_t>(signals.size());
    signals.push_back({n, w, k, idx, 0, false});
    return idx;
}

const Signal* Module::findSignal(const std::string& n) const {
    for (auto& s : signals)
        if (s.name == n) return &s;
    return nullptr;
}

void Module::computeLayout() {
    uint32_t offset = 0;
    for (auto& s : signals) {
        s.stateOffset = offset;
        offset += bytesForWidth(s.width);
    }
    stateSize = offset;
}

} // namespace surge::ir
