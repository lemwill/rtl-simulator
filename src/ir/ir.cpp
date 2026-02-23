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

// ── Dead signal elimination ─────────────────────────────────────────────────

static void collectRefs(const ExprPtr& expr, std::vector<bool>& live) {
    if (!expr) return;
    switch (expr->kind) {
        case ExprKind::SignalRef:
            live[expr->signalIndex] = true;
            break;
        case ExprKind::ArrayElement:
            // All elements in the array are potentially accessed
            for (uint32_t i = 0; i < expr->arraySize; i++)
                live[expr->arrayBaseIndex + i] = true;
            break;
        default:
            break;
    }
    for (auto& op : expr->operands)
        collectRefs(op, live);
}

uint32_t Module::eliminateDeadSignals() {
    const uint32_t N = static_cast<uint32_t>(signals.size());
    if (N == 0) return 0;

    // Step 1: Seed live set with outputs, inputs, and clock signals
    std::vector<bool> live(N, false);
    for (auto& sig : signals) {
        if (sig.kind == SignalKind::Output || sig.kind == SignalKind::Input)
            live[sig.index] = true;
    }
    for (auto& proc : processes) {
        if (proc.kind == Process::Sequential)
            live[proc.clockSignalIndex] = true;
    }

    // Step 2: Build a map from signal index → driving expressions
    // (assignments whose target is this signal)
    std::vector<std::vector<const Assignment*>> drivers(N);
    for (auto& proc : processes) {
        for (auto& assign : proc.assignments) {
            if (assign.arraySize > 0) {
                // Array store: all elements are potential targets
                for (uint32_t i = 0; i < assign.arraySize; i++)
                    drivers[assign.targetIndex + i].push_back(&assign);
            } else {
                drivers[assign.targetIndex].push_back(&assign);
            }
        }
    }

    // Step 3: BFS — propagate liveness through expressions
    std::vector<uint32_t> worklist;
    for (uint32_t i = 0; i < N; i++)
        if (live[i]) worklist.push_back(i);

    while (!worklist.empty()) {
        uint32_t idx = worklist.back();
        worklist.pop_back();
        for (auto* assign : drivers[idx]) {
            auto prevCount = worklist.size();
            // Collect signal refs from value expression
            std::vector<bool> before = live;
            collectRefs(assign->value, live);
            collectRefs(assign->indexExpr, live);
            // Add newly discovered live signals to worklist
            for (uint32_t j = 0; j < N; j++) {
                if (live[j] && !before[j])
                    worklist.push_back(j);
            }
        }
    }

    // Step 4: Remove assignments to dead signals
    uint32_t removed = 0;
    for (auto& proc : processes) {
        auto& assigns = proc.assignments;
        auto newEnd = std::remove_if(assigns.begin(), assigns.end(),
            [&](const Assignment& a) {
                if (a.arraySize > 0) {
                    // Array store: keep if any element is live
                    for (uint32_t i = 0; i < a.arraySize; i++)
                        if (live[a.targetIndex + i]) return false;
                    return true;
                }
                return !live[a.targetIndex];
            });
        removed += static_cast<uint32_t>(assigns.end() - newEnd);
        assigns.erase(newEnd, assigns.end());
    }

    return removed;
}

} // namespace surge::ir
