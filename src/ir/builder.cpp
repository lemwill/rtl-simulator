#include "builder.h"
#include "ir.h"

#include <slang/ast/ASTVisitor.h>
#include <slang/ast/Compilation.h>
#include <slang/ast/Expression.h>
#include <slang/ast/Statement.h>
#include <slang/ast/expressions/AssignmentExpressions.h>
#include <slang/ast/expressions/LiteralExpressions.h>
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/expressions/OperatorExpressions.h>
#include <slang/ast/expressions/ConversionExpression.h>
#include <slang/ast/expressions/Operator.h>
#include <slang/ast/expressions/CallExpression.h>
#include <slang/ast/expressions/SelectExpressions.h>
#include <slang/ast/statements/ConditionalStatements.h>
#include <slang/ast/statements/LoopStatements.h>
#include <slang/ast/statements/MiscStatements.h>
#include <slang/ast/symbols/BlockSymbols.h>
#include <slang/ast/symbols/InstanceSymbols.h>
#include <slang/ast/symbols/MemberSymbols.h>
#include <slang/ast/symbols/PortSymbols.h>
#include <slang/ast/symbols/ParameterSymbols.h>
#include <slang/ast/symbols/VariableSymbols.h>
#include <slang/ast/types/AllTypes.h>
#include <slang/diagnostics/DiagnosticEngine.h>
#include <slang/diagnostics/TextDiagnosticClient.h>
#include <slang/driver/Driver.h>
#include <slang/syntax/SyntaxTree.h>

#include <iostream>
#include <unordered_map>

namespace surge::ir {

namespace {

// ── Helpers ─────────────────────────────────────────────────────────────────

uint32_t getTypeWidth(const slang::ast::Type& type) {
    auto bw = type.getBitWidth();
    return bw > 0 ? static_cast<uint32_t>(bw) : 1;
}

// Unwrap ConversionExpressions to extract a constant integer value.
std::optional<uint64_t> extractConstantInt(const slang::ast::Expression& expr) {
    if (expr.kind == slang::ast::ExpressionKind::IntegerLiteral) {
        return expr.as<slang::ast::IntegerLiteral>().getValue().as<uint64_t>();
    }
    if (expr.kind == slang::ast::ExpressionKind::Conversion) {
        return extractConstantInt(expr.as<slang::ast::ConversionExpression>().operand());
    }
    // Try slang's pre-computed constant value (handles genvar expressions, param arithmetic)
    auto* cv = expr.getConstant();
    if (cv && cv->isInteger())
        return cv->integer().as<uint64_t>();
    return std::nullopt;
}

// ── Array metadata ──────────────────────────────────────────────────────────

struct ArrayInfo {
    uint32_t baseSignalIndex; // signal index of element [0]
    uint32_t size;            // number of elements
    uint32_t elementWidth;    // bit width per element
    int32_t  rangeLower;      // lower bound of the range (e.g., 0 for [0:7])
};

using ArrayMap = std::unordered_map<const slang::ast::Symbol*, ArrayInfo>;
using SymbolMap = std::unordered_map<const slang::ast::Symbol*, uint32_t>;

// Check if a type is a fixed-size unpacked array.
std::optional<std::tuple<uint32_t, uint32_t, int32_t>>
getUnpackedArrayInfo(const slang::ast::Type& type) {
    auto& canon = type.getCanonicalType();
    if (canon.kind == slang::ast::SymbolKind::FixedSizeUnpackedArrayType) {
        auto& arrTy = canon.as<slang::ast::FixedSizeUnpackedArrayType>();
        uint32_t elemWidth = getTypeWidth(arrTy.elementType);
        uint32_t size = static_cast<uint32_t>(arrTy.range.fullWidth());
        int32_t lower = arrTy.range.lower();
        return std::make_tuple(elemWidth, size, lower);
    }
    return std::nullopt;
}

// ── Expression lowering ─────────────────────────────────────────────────────

class ExprLowering {
public:
    ExprLowering(Module& mod, const ArrayMap& arrayMap, const SymbolMap& symbolMap)
        : mod_(mod), arrayMap_(arrayMap), symbolMap_(symbolMap) {}

    void setLoopVarValue(const slang::ast::Symbol* sym, uint64_t val) {
        loopVarValues_[sym] = val;
    }
    void clearLoopVarValue(const slang::ast::Symbol* sym) {
        loopVarValues_.erase(sym);
    }

    ExprPtr lower(const slang::ast::Expression& expr) {
        switch (expr.kind) {
            case slang::ast::ExpressionKind::IntegerLiteral:
                return lowerIntLiteral(expr.as<slang::ast::IntegerLiteral>());
            case slang::ast::ExpressionKind::UnbasedUnsizedIntegerLiteral: {
                auto& lit = expr.as<slang::ast::UnbasedUnsizedIntegerLiteral>();
                auto sv = lit.getValue();
                uint64_t v = 0;
                if (auto opt = sv.as<uint64_t>())
                    v = *opt;
                return Expr::constant(getTypeWidth(*lit.type), v);
            }
            case slang::ast::ExpressionKind::NamedValue:
                return lowerNamedValue(expr.as<slang::ast::NamedValueExpression>());
            case slang::ast::ExpressionKind::UnaryOp:
                return lowerUnary(expr.as<slang::ast::UnaryExpression>());
            case slang::ast::ExpressionKind::BinaryOp:
                return lowerBinary(expr.as<slang::ast::BinaryExpression>());
            case slang::ast::ExpressionKind::ConditionalOp:
                return lowerConditional(expr.as<slang::ast::ConditionalExpression>());
            case slang::ast::ExpressionKind::Conversion:
                return lowerConversion(expr.as<slang::ast::ConversionExpression>());
            case slang::ast::ExpressionKind::Concatenation:
                return lowerConcat(expr.as<slang::ast::ConcatenationExpression>());
            case slang::ast::ExpressionKind::ElementSelect:
                return lowerElementSelect(expr.as<slang::ast::ElementSelectExpression>());
            case slang::ast::ExpressionKind::RangeSelect:
                return lowerRangeSelect(expr.as<slang::ast::RangeSelectExpression>());
            case slang::ast::ExpressionKind::Replication:
                return lowerReplication(expr.as<slang::ast::ReplicationExpression>());
            case slang::ast::ExpressionKind::Call:
                return lowerCall(expr.as<slang::ast::CallExpression>());
            default:
                std::cerr << "surge: unsupported expression kind "
                          << static_cast<int>(expr.kind) << "\n";
                return Expr::constant(1, 0);
        }
    }

    const ArrayMap& arrayMap() const { return arrayMap_; }

private:
    Module& mod_;
    const ArrayMap& arrayMap_;
    const SymbolMap& symbolMap_;
    std::unordered_map<const slang::ast::Symbol*, uint64_t> loopVarValues_;

    ExprPtr lowerIntLiteral(const slang::ast::IntegerLiteral& lit) {
        auto val = lit.getValue();
        uint64_t v = 0;
        if (auto optVal = val.as<uint64_t>())
            v = *optVal;
        return Expr::constant(getTypeWidth(*lit.type), v);
    }

    ExprPtr lowerNamedValue(const slang::ast::NamedValueExpression& nv) {
        auto& sym = nv.symbol;

        // Check if this is a loop variable with a known constant value
        auto lvIt = loopVarValues_.find(&sym);
        if (lvIt != loopVarValues_.end()) {
            return Expr::constant(getTypeWidth(*nv.type), lvIt->second);
        }

        // Handle enum values as constants
        if (sym.kind == slang::ast::SymbolKind::EnumValue) {
            auto& ev = sym.as<slang::ast::EnumValueSymbol>();
            auto& cv = ev.getValue();
            uint64_t v = 0;
            if (cv.isInteger()) {
                if (auto opt = cv.integer().as<uint64_t>())
                    v = *opt;
            }
            return Expr::constant(getTypeWidth(*nv.type), v);
        }

        // Handle parameters/localparams as constants
        if (sym.kind == slang::ast::SymbolKind::Parameter) {
            auto& param = sym.as<slang::ast::ParameterSymbol>();
            auto& cv = param.getValue();
            uint64_t v = 0;
            if (cv.isInteger()) {
                if (auto opt = cv.integer().as<uint64_t>())
                    v = *opt;
            }
            return Expr::constant(getTypeWidth(*nv.type), v);
        }

        // Check if this is an array (whole-array reference — unsupported as expr)
        auto arrIt = arrayMap_.find(&sym);
        if (arrIt != arrayMap_.end()) {
            std::cerr << "surge: unsupported whole-array reference '" << sym.name << "'\n";
            return Expr::constant(1, 0);
        }

        // Symbol-based resolution (supports hierarchical inlining)
        auto symIt = symbolMap_.find(&sym);
        if (symIt != symbolMap_.end()) {
            auto& sig = mod_.signals[symIt->second];
            return Expr::signalRef(sig.width, sig.index);
        }

        // Fallback: name-based resolution
        auto* sig = mod_.findSignal(std::string(sym.name));
        if (!sig) {
            std::cerr << "surge: unknown signal '" << sym.name << "'\n";
            return Expr::constant(1, 0);
        }
        return Expr::signalRef(sig->width, sig->index);
    }

    ExprPtr lowerUnary(const slang::ast::UnaryExpression& ue) {
        auto operand = lower(ue.operand());
        uint32_t w = getTypeWidth(*ue.type);
        UnaryOp op;
        switch (ue.op) {
            case slang::ast::UnaryOperator::BitwiseNot: op = UnaryOp::Not; break;
            case slang::ast::UnaryOperator::LogicalNot: op = UnaryOp::Not; break;
            case slang::ast::UnaryOperator::Minus:      op = UnaryOp::Negate; break;
            case slang::ast::UnaryOperator::Plus:       return operand;
            case slang::ast::UnaryOperator::BitwiseAnd: op = UnaryOp::ReduceAnd; break;
            case slang::ast::UnaryOperator::BitwiseOr:  op = UnaryOp::ReduceOr; break;
            case slang::ast::UnaryOperator::BitwiseXor: op = UnaryOp::ReduceXor; break;
            case slang::ast::UnaryOperator::BitwiseNand:
                return Expr::unary(UnaryOp::Not, w,
                    Expr::unary(UnaryOp::ReduceAnd, 1, operand));
            case slang::ast::UnaryOperator::BitwiseNor:
                return Expr::unary(UnaryOp::Not, w,
                    Expr::unary(UnaryOp::ReduceOr, 1, operand));
            case slang::ast::UnaryOperator::BitwiseXnor:
                return Expr::unary(UnaryOp::Not, w,
                    Expr::unary(UnaryOp::ReduceXor, 1, operand));
            default:
                std::cerr << "surge: unsupported unary op "
                          << static_cast<int>(ue.op) << "\n";
                return operand;
        }
        return Expr::unary(op, w, operand);
    }

    ExprPtr lowerBinary(const slang::ast::BinaryExpression& be) {
        auto lhs = lower(be.left());
        auto rhs = lower(be.right());
        uint32_t w = getTypeWidth(*be.type);
        BinaryOp op;
        switch (be.op) {
            case slang::ast::BinaryOperator::Add:           op = BinaryOp::Add; break;
            case slang::ast::BinaryOperator::Subtract:      op = BinaryOp::Sub; break;
            case slang::ast::BinaryOperator::Multiply:      op = BinaryOp::Mul; break;
            case slang::ast::BinaryOperator::BinaryAnd:     op = BinaryOp::And; break;
            case slang::ast::BinaryOperator::BinaryOr:      op = BinaryOp::Or; break;
            case slang::ast::BinaryOperator::BinaryXor:     op = BinaryOp::Xor; break;
            case slang::ast::BinaryOperator::Equality:      op = BinaryOp::Eq; break;
            case slang::ast::BinaryOperator::Inequality:    op = BinaryOp::Neq; break;
            case slang::ast::BinaryOperator::LessThan:      op = BinaryOp::Lt; break;
            case slang::ast::BinaryOperator::LessThanEqual: op = BinaryOp::Lte; break;
            case slang::ast::BinaryOperator::GreaterThan:   op = BinaryOp::Gt; break;
            case slang::ast::BinaryOperator::GreaterThanEqual: op = BinaryOp::Gte; break;
            case slang::ast::BinaryOperator::LogicalShiftLeft:  op = BinaryOp::Shl; break;
            case slang::ast::BinaryOperator::LogicalShiftRight: op = BinaryOp::Shr; break;
            case slang::ast::BinaryOperator::ArithmeticShiftLeft:  op = BinaryOp::Shl; break;
            case slang::ast::BinaryOperator::ArithmeticShiftRight:
                op = be.left().type->isSigned() ? BinaryOp::AShr : BinaryOp::Shr;
                break;
            case slang::ast::BinaryOperator::Divide:       op = BinaryOp::Div; break;
            case slang::ast::BinaryOperator::Mod:          op = BinaryOp::Mod; break;
            case slang::ast::BinaryOperator::BinaryXnor:
                return Expr::unary(UnaryOp::Not, w, Expr::binary(BinaryOp::Xor, w, lhs, rhs));
            case slang::ast::BinaryOperator::LogicalAnd: {
                auto lb = Expr::binary(BinaryOp::Neq, 1, lhs,
                    Expr::constant(getTypeWidth(*be.left().type), 0));
                auto rb = Expr::binary(BinaryOp::Neq, 1, rhs,
                    Expr::constant(getTypeWidth(*be.right().type), 0));
                return Expr::binary(BinaryOp::And, w, lb, rb);
            }
            case slang::ast::BinaryOperator::LogicalOr: {
                auto lb = Expr::binary(BinaryOp::Neq, 1, lhs,
                    Expr::constant(getTypeWidth(*be.left().type), 0));
                auto rb = Expr::binary(BinaryOp::Neq, 1, rhs,
                    Expr::constant(getTypeWidth(*be.right().type), 0));
                return Expr::binary(BinaryOp::Or, w, lb, rb);
            }
            default:
                std::cerr << "surge: unsupported binary op "
                          << static_cast<int>(be.op) << "\n";
                return lhs;
        }
        auto result = Expr::binary(op, w, lhs, rhs);
        // Propagate signedness for comparison and arithmetic ops
        if (be.left().type->isSigned() && be.right().type->isSigned())
            result->isSigned = true;
        return result;
    }

    ExprPtr lowerConditional(const slang::ast::ConditionalExpression& ce) {
        auto& conditions = ce.conditions;
        ExprPtr cond;
        if (!conditions.empty())
            cond = lower(*conditions[0].expr);
        else
            cond = Expr::constant(1, 1);

        auto t = lower(ce.left());
        auto f = lower(ce.right());
        return Expr::mux(getTypeWidth(*ce.type), cond, t, f);
    }

    ExprPtr lowerConversion(const slang::ast::ConversionExpression& cv) {
        auto inner = lower(cv.operand());
        // If the operand is signed and we're widening, we need sign-extension.
        // Mark the expression so codegen can use sext instead of zext.
        if (cv.operand().type->isSigned()) {
            inner->isSigned = true;
        }
        return inner;
    }

    ExprPtr lowerConcat(const slang::ast::ConcatenationExpression& cc) {
        std::vector<ExprPtr> parts;
        for (auto* op : cc.operands())
            parts.push_back(lower(*op));
        return Expr::concat(getTypeWidth(*cc.type), std::move(parts));
    }

    ExprPtr lowerReplication(const slang::ast::ReplicationExpression& re) {
        auto count = extractConstantInt(re.count());
        if (!count || *count == 0)
            return Expr::constant(1, 0);
        auto inner = lower(re.concat());
        uint32_t totalWidth = getTypeWidth(*re.type);
        std::vector<ExprPtr> parts;
        for (uint64_t i = 0; i < *count; i++)
            parts.push_back(inner);
        return Expr::concat(totalWidth, std::move(parts));
    }

    ExprPtr lowerCall(const slang::ast::CallExpression& call) {
        if (call.isSystemCall()) {
            auto name = call.getSubroutineName();
            auto args = call.arguments();

            // $signed / $unsigned — pass through with signedness flag
            if ((name == "$signed" || name == "$unsigned") && args.size() == 1) {
                auto inner = lower(*args[0]);
                if (name == "$signed")
                    inner->isSigned = true;
                else
                    inner->isSigned = false;
                return inner;
            }

            // $countones — population count
            if (name == "$countones" && args.size() == 1) {
                auto inner = lower(*args[0]);
                return Expr::unary(UnaryOp::Popcount, getTypeWidth(*call.type), inner);
            }

            // $onehot — exactly one bit set (popcount == 1)
            if (name == "$onehot" && args.size() == 1) {
                auto inner = lower(*args[0]);
                auto pop = Expr::unary(UnaryOp::Popcount, inner->width, inner);
                return Expr::binary(BinaryOp::Eq, 1, pop,
                    Expr::constant(inner->width, 1));
            }

            // $onehot0 — zero or one bit set (popcount <= 1)
            if (name == "$onehot0" && args.size() == 1) {
                auto inner = lower(*args[0]);
                auto pop = Expr::unary(UnaryOp::Popcount, inner->width, inner);
                return Expr::binary(BinaryOp::Lte, 1, pop,
                    Expr::constant(inner->width, 1));
            }

            // $isunknown — always 0 in 2-state simulation
            if (name == "$isunknown" && args.size() == 1) {
                return Expr::constant(1, 0);
            }

            // $clog2, $bits — try compile-time constant evaluation
            if (auto* cv = call.getConstant()) {
                if (cv->isInteger()) {
                    uint64_t v = 0;
                    if (auto opt = cv->integer().as<uint64_t>())
                        v = *opt;
                    return Expr::constant(getTypeWidth(*call.type), v);
                }
            }
        }
        std::cerr << "surge: unsupported call: " << call.getSubroutineName() << "\n";
        return Expr::constant(getTypeWidth(*call.type), 0);
    }

    ExprPtr lowerElementSelect(const slang::ast::ElementSelectExpression& es) {
        // Check if this is an unpacked array element access
        auto& baseType = es.value().type->getCanonicalType();
        if (baseType.kind == slang::ast::SymbolKind::FixedSizeUnpackedArrayType) {
            return lowerUnpackedArraySelect(es);
        }

        // Packed bit select
        auto src = lower(es.value());
        auto idx = extractConstantInt(es.selector());
        if (idx) {
            uint32_t bit = static_cast<uint32_t>(*idx);
            return Expr::slice(1, src, bit, bit);
        }
        // Dynamic index: (src >> sel) & 1
        auto selExpr = lower(es.selector());
        uint32_t srcWidth = getTypeWidth(*es.value().type);
        auto shifted = Expr::binary(BinaryOp::Shr, srcWidth, src, selExpr);
        return Expr::binary(BinaryOp::And, 1, shifted, Expr::constant(srcWidth, 1));
    }

    ExprPtr lowerUnpackedArraySelect(const slang::ast::ElementSelectExpression& es) {
        // Get the array base symbol
        const slang::ast::Symbol* baseSym = nullptr;
        if (es.value().kind == slang::ast::ExpressionKind::NamedValue) {
            baseSym = &es.value().as<slang::ast::NamedValueExpression>().symbol;
        }

        if (!baseSym) {
            std::cerr << "surge: unsupported unpacked array base expression\n";
            return Expr::constant(getTypeWidth(*es.type), 0);
        }

        auto arrIt = arrayMap_.find(baseSym);
        if (arrIt == arrayMap_.end()) {
            std::cerr << "surge: array info not found for '" << baseSym->name << "'\n";
            return Expr::constant(getTypeWidth(*es.type), 0);
        }

        auto& info = arrIt->second;

        // Check for constant index
        auto constIdx = extractConstantInt(es.selector());
        if (constIdx) {
            int64_t idx = static_cast<int64_t>(*constIdx) - info.rangeLower;
            if (idx < 0 || static_cast<uint32_t>(idx) >= info.size) {
                std::cerr << "surge: array index out of bounds: " << *constIdx << "\n";
                return Expr::constant(info.elementWidth, 0);
            }
            uint32_t sigIdx = info.baseSignalIndex + static_cast<uint32_t>(idx);
            return Expr::signalRef(info.elementWidth, sigIdx);
        }

        // Dynamic index: emit ArrayElement expression
        auto indexExpr = lower(es.selector());
        // Subtract range lower bound if non-zero
        if (info.rangeLower != 0) {
            indexExpr = Expr::binary(BinaryOp::Sub, indexExpr->width, indexExpr,
                Expr::constant(indexExpr->width, static_cast<uint64_t>(
                    static_cast<int64_t>(info.rangeLower))));
        }
        return Expr::arrayElement(info.elementWidth, info.baseSignalIndex, info.size, indexExpr);
    }

    ExprPtr lowerRangeSelect(const slang::ast::RangeSelectExpression& rs) {
        auto src = lower(rs.value());
        uint32_t resultWidth = getTypeWidth(*rs.type);
        uint32_t srcWidth = getTypeWidth(*rs.value().type);
        auto selKind = rs.getSelectionKind();

        auto leftVal = extractConstantInt(rs.left());
        auto rightVal = extractConstantInt(rs.right());

        if (selKind == slang::ast::RangeSelectionKind::Simple && leftVal && rightVal) {
            uint32_t hi = static_cast<uint32_t>(*leftVal);
            uint32_t lo = static_cast<uint32_t>(*rightVal);
            return Expr::slice(resultWidth, src, hi, lo);
        }
        if (selKind == slang::ast::RangeSelectionKind::IndexedUp && leftVal) {
            uint32_t base = static_cast<uint32_t>(*leftVal);
            return Expr::slice(resultWidth, src, base + resultWidth - 1, base);
        }
        if (selKind == slang::ast::RangeSelectionKind::IndexedDown && leftVal) {
            uint32_t base = static_cast<uint32_t>(*leftVal);
            if (base + 1 < resultWidth) {
                std::cerr << "surge: IndexedDown range select underflow\n";
                return Expr::constant(resultWidth, 0);
            }
            return Expr::slice(resultWidth, src, base, base - resultWidth + 1);
        }

        // Dynamic range select: start is a runtime expression, width is constant
        if (selKind == slang::ast::RangeSelectionKind::IndexedUp) {
            // sig[start+:WIDTH] → (src >> start) truncated to WIDTH bits
            auto startExpr = lower(rs.left());
            auto shifted = Expr::binary(BinaryOp::Shr, srcWidth, src, startExpr);
            uint64_t mask = (resultWidth >= 64) ? ~0ULL : ((1ULL << resultWidth) - 1);
            return Expr::binary(BinaryOp::And, resultWidth, shifted,
                Expr::constant(srcWidth, mask));
        }
        if (selKind == slang::ast::RangeSelectionKind::IndexedDown) {
            // sig[start-:WIDTH] → (src >> (start - WIDTH + 1)) truncated to WIDTH bits
            auto startExpr = lower(rs.left());
            auto base = Expr::binary(BinaryOp::Sub, startExpr->width, startExpr,
                Expr::constant(startExpr->width, resultWidth - 1));
            auto shifted = Expr::binary(BinaryOp::Shr, srcWidth, src, base);
            uint64_t mask = (resultWidth >= 64) ? ~0ULL : ((1ULL << resultWidth) - 1);
            return Expr::binary(BinaryOp::And, resultWidth, shifted,
                Expr::constant(srcWidth, mask));
        }

        std::cerr << "surge: unsupported range select (dynamic bounds)\n";
        return Expr::constant(resultWidth, 0);
    }
};

// ── For-loop helpers ────────────────────────────────────────────────────────

// Unwrap NamedValue from possible Conversion wrappers.
const slang::ast::Symbol* unwrapNamedSymbol(const slang::ast::Expression& expr) {
    if (expr.kind == slang::ast::ExpressionKind::NamedValue)
        return &expr.as<slang::ast::NamedValueExpression>().symbol;
    if (expr.kind == slang::ast::ExpressionKind::Conversion)
        return unwrapNamedSymbol(expr.as<slang::ast::ConversionExpression>().operand());
    return nullptr;
}

int64_t detectStepIncrement(const slang::ast::Expression& stepExpr,
                            const slang::ast::Symbol* loopVar) {
    if (stepExpr.kind == slang::ast::ExpressionKind::Assignment) {
        auto& assign = stepExpr.as<slang::ast::AssignmentExpression>();
        // Compound assignment: i += N or i -= N
        if (assign.isCompound()) {
            auto val = extractConstantInt(assign.right());
            if (assign.op.value() == slang::ast::BinaryOperator::Add)
                return val ? static_cast<int64_t>(*val) : 1;
            if (assign.op.value() == slang::ast::BinaryOperator::Subtract)
                return val ? -static_cast<int64_t>(*val) : -1;
        }
        // i = i + N pattern
        if (assign.right().kind == slang::ast::ExpressionKind::BinaryOp) {
            auto& bin = assign.right().as<slang::ast::BinaryExpression>();
            if (bin.op == slang::ast::BinaryOperator::Add) {
                auto rval = extractConstantInt(bin.right());
                if (rval) return static_cast<int64_t>(*rval);
                auto lval = extractConstantInt(bin.left());
                if (lval) return static_cast<int64_t>(*lval);
            }
            if (bin.op == slang::ast::BinaryOperator::Subtract) {
                auto rval = extractConstantInt(bin.right());
                if (rval) return -static_cast<int64_t>(*rval);
            }
        }
    }
    // Check if wrapped in conversion
    if (stepExpr.kind == slang::ast::ExpressionKind::Conversion) {
        return detectStepIncrement(
            stepExpr.as<slang::ast::ConversionExpression>().operand(), loopVar);
    }
    return 1; // default
}

bool evaluateStopCondition(const slang::ast::Expression& stopExpr,
                           const slang::ast::Symbol* loopVar,
                           uint64_t currentVal) {
    if (stopExpr.kind == slang::ast::ExpressionKind::BinaryOp) {
        auto& bin = stopExpr.as<slang::ast::BinaryExpression>();
        uint64_t bound = 0;
        bool varOnLeft = false;

        auto* leftSym = unwrapNamedSymbol(bin.left());
        auto* rightSym = unwrapNamedSymbol(bin.right());

        if (leftSym == loopVar) {
            varOnLeft = true;
            auto val = extractConstantInt(bin.right());
            if (!val) return false;
            bound = *val;
        } else if (rightSym == loopVar) {
            auto val = extractConstantInt(bin.left());
            if (!val) return false;
            bound = *val;
        } else {
            return false;
        }

        int64_t sv = static_cast<int64_t>(currentVal);
        int64_t sb = static_cast<int64_t>(bound);

        switch (bin.op) {
            case slang::ast::BinaryOperator::LessThan:
                return varOnLeft ? (sv < sb) : (sb < sv);
            case slang::ast::BinaryOperator::LessThanEqual:
                return varOnLeft ? (sv <= sb) : (sb <= sv);
            case slang::ast::BinaryOperator::GreaterThan:
                return varOnLeft ? (sv > sb) : (sb > sv);
            case slang::ast::BinaryOperator::GreaterThanEqual:
                return varOnLeft ? (sv >= sb) : (sb >= sv);
            case slang::ast::BinaryOperator::Inequality:
                return currentVal != bound;
            case slang::ast::BinaryOperator::Equality:
                return currentVal == bound;
            default: break;
        }
    }
    if (stopExpr.kind == slang::ast::ExpressionKind::Conversion) {
        return evaluateStopCondition(
            stopExpr.as<slang::ast::ConversionExpression>().operand(), loopVar, currentVal);
    }
    return false;
}

// ── Statement lowering (builds assignments) ─────────────────────────────────

class StmtLowering {
public:
    StmtLowering(Module& mod, ExprLowering& el, const SymbolMap& symbolMap)
        : mod_(mod), el_(el), symbolMap_(symbolMap) {}

    void lower(const slang::ast::Statement& stmt) {
        switch (stmt.kind) {
            case slang::ast::StatementKind::ExpressionStatement:
                lowerExprStmt(stmt.as<slang::ast::ExpressionStatement>());
                break;
            case slang::ast::StatementKind::Conditional:
                lowerConditional(stmt.as<slang::ast::ConditionalStatement>());
                break;
            case slang::ast::StatementKind::Block:
                lowerBlock(stmt.as<slang::ast::BlockStatement>());
                break;
            case slang::ast::StatementKind::List:
                lowerList(stmt.as<slang::ast::StatementList>());
                break;
            case slang::ast::StatementKind::Timed:
                // Skip timing control (e.g. @(posedge clk)), lower the body
                lower(stmt.as<slang::ast::TimedStatement>().stmt);
                break;
            case slang::ast::StatementKind::Case:
                lowerCase(stmt.as<slang::ast::CaseStatement>());
                break;
            case slang::ast::StatementKind::ForLoop:
                lowerForLoop(stmt.as<slang::ast::ForLoopStatement>());
                break;
            case slang::ast::StatementKind::VariableDeclaration:
                // Skip variable declarations (e.g., loop variable `int i`)
                break;
            default:
                std::cerr << "surge: unsupported statement kind "
                          << static_cast<int>(stmt.kind) << "\n";
                break;
        }
    }

    std::vector<Assignment>& assignments() { return assignments_; }

private:
    Module& mod_;
    ExprLowering& el_;
    const SymbolMap& symbolMap_;
    std::vector<Assignment> assignments_;

    void lowerExprStmt(const slang::ast::ExpressionStatement& es) {
        auto& expr = es.expr;
        if (expr.kind == slang::ast::ExpressionKind::Assignment) {
            auto& assign = expr.as<slang::ast::AssignmentExpression>();
            lowerAssignment(assign);
        }
    }

    uint32_t resolveSignalIndex(const slang::ast::Symbol& sym) const {
        auto it = symbolMap_.find(&sym);
        if (it != symbolMap_.end()) return it->second;
        auto* sig = mod_.findSignal(std::string(sym.name));
        if (sig) return sig->index;
        return UINT32_MAX;
    }

    void lowerAssignment(const slang::ast::AssignmentExpression& ae) {
        auto& lhs = ae.left();
        auto rhs = el_.lower(ae.right());

        if (lhs.kind == slang::ast::ExpressionKind::NamedValue) {
            auto& nv = lhs.as<slang::ast::NamedValueExpression>();
            uint32_t sigIdx = resolveSignalIndex(nv.symbol);
            if (sigIdx == UINT32_MAX) {
                std::cerr << "surge: unknown target signal '" << nv.symbol.name << "'\n";
                return;
            }
            assignments_.push_back({sigIdx, rhs});
        } else if (lhs.kind == slang::ast::ExpressionKind::ElementSelect) {
            lowerIndexedAssignment(lhs.as<slang::ast::ElementSelectExpression>(), rhs);
        } else if (lhs.kind == slang::ast::ExpressionKind::RangeSelect) {
            lowerRangeAssignment(lhs.as<slang::ast::RangeSelectExpression>(), rhs);
        } else {
            std::cerr << "surge: unsupported LHS kind "
                      << static_cast<int>(lhs.kind) << " in assignment\n";
        }
    }

    void lowerIndexedAssignment(const slang::ast::ElementSelectExpression& es, ExprPtr rhs) {
        // Get array base symbol
        const slang::ast::Symbol* baseSym = nullptr;
        if (es.value().kind == slang::ast::ExpressionKind::NamedValue) {
            baseSym = &es.value().as<slang::ast::NamedValueExpression>().symbol;
        }

        if (!baseSym) {
            std::cerr << "surge: unsupported indexed LHS base\n";
            return;
        }

        auto& arrayMap = el_.arrayMap();
        auto arrIt = arrayMap.find(baseSym);
        if (arrIt == arrayMap.end()) {
            // Packed bit assignment: sig[idx] <= rhs
            // Becomes: sig <= (sig & ~(1 << idx)) | ((rhs & 1) << idx)
            uint32_t sigIdx = resolveSignalIndex(*baseSym);
            if (sigIdx == UINT32_MAX) {
                std::cerr << "surge: unknown signal for packed bit assign '"
                          << baseSym->name << "'\n";
                return;
            }
            auto& sig = mod_.signals[sigIdx];
            auto current = el_.lower(es.value()); // current signal value
            auto one = Expr::constant(sig.width, 1);

            auto constIdx = extractConstantInt(es.selector());
            if (constIdx) {
                uint32_t bit = static_cast<uint32_t>(*constIdx);
                uint64_t mask = ~(uint64_t(1) << bit);
                auto cleared = Expr::binary(BinaryOp::And, sig.width, current,
                    Expr::constant(sig.width, mask));
                auto rhsBit = Expr::binary(BinaryOp::And, sig.width, rhs, one);
                auto shifted = Expr::binary(BinaryOp::Shl, sig.width, rhsBit,
                    Expr::constant(sig.width, bit));
                assignments_.push_back({sigIdx, Expr::binary(BinaryOp::Or, sig.width, cleared, shifted)});
            } else {
                auto selExpr = el_.lower(es.selector());
                auto mask = Expr::unary(UnaryOp::Not, sig.width,
                    Expr::binary(BinaryOp::Shl, sig.width, one, selExpr));
                auto cleared = Expr::binary(BinaryOp::And, sig.width, current, mask);
                auto rhsBit = Expr::binary(BinaryOp::And, sig.width, rhs, one);
                auto shifted = Expr::binary(BinaryOp::Shl, sig.width, rhsBit, selExpr);
                assignments_.push_back({sigIdx, Expr::binary(BinaryOp::Or, sig.width, cleared, shifted)});
            }
            return;
        }

        auto& info = arrIt->second;

        // Check for constant index
        auto constIdx = extractConstantInt(es.selector());
        if (constIdx) {
            int64_t idx = static_cast<int64_t>(*constIdx) - info.rangeLower;
            if (idx < 0 || static_cast<uint32_t>(idx) >= info.size) {
                std::cerr << "surge: array assignment index out of bounds\n";
                return;
            }
            uint32_t sigIdx = info.baseSignalIndex + static_cast<uint32_t>(idx);
            assignments_.push_back({sigIdx, rhs});
            return;
        }

        // Try lowering the selector — it may resolve to a constant (e.g., loop variable)
        auto selExpr = el_.lower(es.selector());
        // Subtract range lower bound if non-zero
        if (info.rangeLower != 0) {
            selExpr = Expr::binary(BinaryOp::Sub, selExpr->width, selExpr,
                Expr::constant(selExpr->width, static_cast<uint64_t>(
                    static_cast<int64_t>(info.rangeLower))));
        }

        // Check if selector resolved to a constant after lowering
        if (selExpr->kind == ExprKind::Constant) {
            uint32_t idx = static_cast<uint32_t>(selExpr->constVal);
            if (idx < info.size) {
                assignments_.push_back({info.baseSignalIndex + idx, rhs});
            } else {
                std::cerr << "surge: array assignment index out of bounds (resolved constant)\n";
            }
            return;
        }

        // Dynamic index: emit a single computed array store
        Assignment a;
        a.targetIndex  = info.baseSignalIndex;
        a.value        = rhs;
        a.indexExpr    = selExpr;
        a.arraySize    = info.size;
        a.elementWidth = info.elementWidth;
        assignments_.push_back(std::move(a));
    }

    void lowerRangeAssignment(const slang::ast::RangeSelectExpression& rs, ExprPtr rhs) {
        // Packed range assignment: sig[hi:lo] <= rhs or sig[start+:W] <= rhs
        // Becomes: sig <= (sig & ~(mask << lo)) | ((rhs & mask) << lo)
        const slang::ast::Symbol* baseSym = nullptr;
        if (rs.value().kind == slang::ast::ExpressionKind::NamedValue) {
            baseSym = &rs.value().as<slang::ast::NamedValueExpression>().symbol;
        }
        if (!baseSym) {
            std::cerr << "surge: unsupported range assignment base\n";
            return;
        }
        uint32_t sigIdx = resolveSignalIndex(*baseSym);
        if (sigIdx == UINT32_MAX) {
            std::cerr << "surge: unknown signal for range assign '" << baseSym->name << "'\n";
            return;
        }
        auto& sig = mod_.signals[sigIdx];
        auto current = el_.lower(rs.value()); // current signal value
        uint32_t rangeWidth = getTypeWidth(*rs.type);
        uint64_t rangeMask = (rangeWidth >= 64) ? ~0ULL : ((1ULL << rangeWidth) - 1);

        auto selKind = rs.getSelectionKind();
        auto leftVal = extractConstantInt(rs.left());
        auto rightVal = extractConstantInt(rs.right());

        ExprPtr loExpr;
        if (selKind == slang::ast::RangeSelectionKind::Simple && leftVal && rightVal) {
            loExpr = Expr::constant(sig.width, *rightVal);
        } else if (selKind == slang::ast::RangeSelectionKind::IndexedUp) {
            // sig[start+:W]: lo = start
            if (leftVal)
                loExpr = Expr::constant(sig.width, *leftVal);
            else
                loExpr = el_.lower(rs.left());
        } else if (selKind == slang::ast::RangeSelectionKind::IndexedDown) {
            // sig[start-:W]: lo = start - W + 1
            if (leftVal)
                loExpr = Expr::constant(sig.width, *leftVal - rangeWidth + 1);
            else
                loExpr = Expr::binary(BinaryOp::Sub, sig.width,
                    el_.lower(rs.left()),
                    Expr::constant(sig.width, rangeWidth - 1));
        } else {
            std::cerr << "surge: unsupported range assignment kind\n";
            return;
        }

        // Build: sig = (sig & ~(mask << lo)) | ((rhs & mask) << lo)
        auto mask = Expr::constant(sig.width, rangeMask);
        auto shiftedMask = Expr::binary(BinaryOp::Shl, sig.width, mask, loExpr);
        auto invMask = Expr::unary(UnaryOp::Not, sig.width, shiftedMask);
        auto cleared = Expr::binary(BinaryOp::And, sig.width, current, invMask);
        auto maskedRhs = Expr::binary(BinaryOp::And, sig.width, rhs, mask);
        auto shiftedRhs = Expr::binary(BinaryOp::Shl, sig.width, maskedRhs, loExpr);
        auto result = Expr::binary(BinaryOp::Or, sig.width, cleared, shiftedRhs);
        assignments_.push_back({sigIdx, result});
    }

    void lowerForLoop(const slang::ast::ForLoopStatement& fl) {
        const slang::ast::Symbol* loopVar = nullptr;
        uint64_t currentVal = 0;

        // Extract loop variable and initial value
        if (!fl.loopVars.empty()) {
            loopVar = fl.loopVars[0];
            if (loopVar->kind == slang::ast::SymbolKind::Variable) {
                if (auto* init = loopVar->as<slang::ast::VariableSymbol>().getInitializer()) {
                    auto val = extractConstantInt(*init);
                    if (val) currentVal = *val;
                }
            }
        } else if (!fl.initializers.empty()) {
            auto& initExpr = *fl.initializers[0];
            // Unwrap possible conversion
            const slang::ast::Expression* inner = &initExpr;
            if (inner->kind == slang::ast::ExpressionKind::Conversion)
                inner = &inner->as<slang::ast::ConversionExpression>().operand();
            if (inner->kind == slang::ast::ExpressionKind::Assignment) {
                auto& assign = inner->as<slang::ast::AssignmentExpression>();
                auto* sym = unwrapNamedSymbol(assign.left());
                if (sym) loopVar = sym;
                auto val = extractConstantInt(assign.right());
                if (val) currentVal = *val;
            }
        }

        if (!loopVar) {
            std::cerr << "surge: for loop has no identifiable loop variable\n";
            return;
        }

        // Determine step increment
        int64_t stepIncrement = 1;
        if (!fl.steps.empty()) {
            stepIncrement = detectStepIncrement(*fl.steps[0], loopVar);
        }

        // Unroll loop (compile-time constant bounds only)
        const uint32_t MAX_UNROLL = 1024;
        uint32_t iter;
        for (iter = 0; iter < MAX_UNROLL; iter++) {
            if (fl.stopExpr) {
                if (!evaluateStopCondition(*fl.stopExpr, loopVar, currentVal))
                    break;
            }

            el_.setLoopVarValue(loopVar, currentVal);
            lower(fl.body);

            currentVal = static_cast<uint64_t>(static_cast<int64_t>(currentVal) + stepIncrement);
        }

        el_.clearLoopVarValue(loopVar);

        if (iter == MAX_UNROLL)
            std::cerr << "surge: for loop unroll limit reached (" << MAX_UNROLL << ")\n";
    }

    // Partition array-store assignments from scalar assignments
    static std::vector<Assignment> extractArrayStores(std::vector<Assignment>& assigns) {
        std::vector<Assignment> arrayStores;
        std::vector<Assignment> scalar;
        for (auto& a : assigns) {
            if (a.indexExpr)
                arrayStores.push_back(std::move(a));
            else
                scalar.push_back(std::move(a));
        }
        assigns = std::move(scalar);
        return arrayStores;
    }

    void lowerConditional(const slang::ast::ConditionalStatement& cs) {
        auto& conditions = cs.conditions;
        ExprPtr cond;
        if (!conditions.empty())
            cond = el_.lower(*conditions[0].expr);
        else
            cond = Expr::constant(1, 1);

        StmtLowering ifLower(mod_, el_, symbolMap_);
        ifLower.lower(cs.ifTrue);
        auto ifAssigns = std::move(ifLower.assignments());

        std::vector<Assignment> elseAssigns;
        if (cs.ifFalse) {
            StmtLowering elseLower(mod_, el_, symbolMap_);
            elseLower.lower(*cs.ifFalse);
            elseAssigns = std::move(elseLower.assignments());
        }

        // Separate array stores from scalar assignments before merging
        auto ifArrayStores = extractArrayStores(ifAssigns);
        auto elseArrayStores = extractArrayStores(elseAssigns);

        // Merge scalar assignments with mux (existing logic)
        for (auto& ifA : ifAssigns) {
            ExprPtr falseVal;
            for (auto& elseA : elseAssigns) {
                if (elseA.targetIndex == ifA.targetIndex) {
                    falseVal = elseA.value;
                    break;
                }
            }
            if (!falseVal) {
                auto& sig = mod_.signals[ifA.targetIndex];
                falseVal = Expr::signalRef(sig.width, sig.index);
            }
            auto muxed = Expr::mux(mod_.signals[ifA.targetIndex].width,
                                   cond, ifA.value, falseVal);
            assignments_.push_back({ifA.targetIndex, muxed});
        }

        for (auto& elseA : elseAssigns) {
            bool found = false;
            for (auto& ifA : ifAssigns) {
                if (ifA.targetIndex == elseA.targetIndex) { found = true; break; }
            }
            if (!found) {
                auto& sig = mod_.signals[elseA.targetIndex];
                auto trueVal = Expr::signalRef(sig.width, sig.index);
                auto muxed = Expr::mux(sig.width, cond, trueVal, elseA.value);
                assignments_.push_back({elseA.targetIndex, muxed});
            }
        }

        // Handle array stores: wrap value in condition mux, preserve array metadata
        for (auto& as : ifArrayStores) {
            auto currentElem = Expr::arrayElement(as.elementWidth, as.targetIndex,
                                                   as.arraySize, as.indexExpr);
            as.value = Expr::mux(as.elementWidth, cond, as.value, currentElem);
            assignments_.push_back(std::move(as));
        }
        for (auto& as : elseArrayStores) {
            auto currentElem = Expr::arrayElement(as.elementWidth, as.targetIndex,
                                                   as.arraySize, as.indexExpr);
            as.value = Expr::mux(as.elementWidth, cond, currentElem, as.value);
            assignments_.push_back(std::move(as));
        }
    }

    void lowerBlock(const slang::ast::BlockStatement& bs) {
        lower(bs.body);
    }

    void lowerList(const slang::ast::StatementList& sl) {
        for (auto* s : sl.list)
            lower(*s);
    }

    void lowerCase(const slang::ast::CaseStatement& cs) {
        auto selectorExpr = el_.lower(cs.expr);

        // Collect default assignments (scalar only; array stores handled separately)
        std::unordered_map<uint32_t, ExprPtr> currentValues;
        std::vector<Assignment> arrayStoreAccum; // accumulated array stores with conditions
        if (cs.defaultCase) {
            StmtLowering defaultLower(mod_, el_, symbolMap_);
            defaultLower.lower(*cs.defaultCase);
            auto defaultAssigns = std::move(defaultLower.assignments());
            auto defaultArrayStores = extractArrayStores(defaultAssigns);
            for (auto& a : defaultAssigns)
                currentValues[a.targetIndex] = a.value;
            // Default array stores get no condition wrapping — they are the fallback
            for (auto& as : defaultArrayStores)
                arrayStoreAccum.push_back(std::move(as));
        }

        // Process items in reverse for correct priority (first match wins)
        for (int i = static_cast<int>(cs.items.size()) - 1; i >= 0; i--) {
            auto& item = cs.items[i];

            StmtLowering itemLower(mod_, el_, symbolMap_);
            itemLower.lower(*item.stmt);
            auto itemAssigns = std::move(itemLower.assignments());
            auto itemArrayStores = extractArrayStores(itemAssigns);

            // Build condition: selector == expr1 || selector == expr2 || ...
            ExprPtr cond;
            for (auto* matchExpr : item.expressions) {
                auto matchVal = el_.lower(*matchExpr);
                auto eq = Expr::binary(BinaryOp::Eq, 1, selectorExpr, matchVal);
                if (!cond)
                    cond = eq;
                else
                    cond = Expr::binary(BinaryOp::Or, 1, cond, eq);
            }
            if (!cond)
                cond = Expr::constant(1, 0);

            // Wrap scalar assignments in a mux
            for (auto& ia : itemAssigns) {
                ExprPtr falseVal;
                auto it = currentValues.find(ia.targetIndex);
                if (it != currentValues.end()) {
                    falseVal = it->second;
                } else {
                    auto& sig = mod_.signals[ia.targetIndex];
                    falseVal = Expr::signalRef(sig.width, sig.index);
                }
                auto muxed = Expr::mux(mod_.signals[ia.targetIndex].width,
                                       cond, ia.value, falseVal);
                currentValues[ia.targetIndex] = muxed;
            }

            // Wrap array stores in a condition mux
            for (auto& as : itemArrayStores) {
                auto currentElem = Expr::arrayElement(as.elementWidth, as.targetIndex,
                                                       as.arraySize, as.indexExpr);
                as.value = Expr::mux(as.elementWidth, cond, as.value, currentElem);
                arrayStoreAccum.push_back(std::move(as));
            }
        }

        for (auto& [targetIdx, val] : currentValues)
            assignments_.push_back({targetIdx, val});
        for (auto& as : arrayStoreAccum)
            assignments_.push_back(std::move(as));
    }
};

} // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

std::unique_ptr<Module> buildFromFile(const std::string& path) {
    auto tree = slang::syntax::SyntaxTree::fromFile(path);
    if (!tree) {
        std::cerr << "surge: failed to parse '" << path << "'\n";
        return nullptr;
    }

    slang::ast::Compilation compilation;
    compilation.addSyntaxTree(tree.value());

    auto diags = compilation.getAllDiagnostics();
    if (!diags.empty()) {
        auto client = std::make_shared<slang::TextDiagnosticClient>();
        client->showColors(false);
        slang::DiagnosticEngine diagEngine(*compilation.getSourceManager());
        diagEngine.addClient(client);
        for (auto& d : diags)
            diagEngine.issue(d);
        std::string diagStr = client->getString();
        if (!diagStr.empty())
            std::cerr << diagStr;
    }

    auto& root = compilation.getRoot();
    const slang::ast::InstanceSymbol* topInst = nullptr;
    for (auto& member : root.members()) {
        if (member.kind == slang::ast::SymbolKind::Instance) {
            topInst = &member.as<slang::ast::InstanceSymbol>();
            break;
        }
    }

    if (!topInst) {
        std::cerr << "surge: no module instance found\n";
        return nullptr;
    }

    auto mod = std::make_unique<Module>();
    mod->name = std::string(topInst->name);

    // Shared maps for all scopes
    ArrayMap arrayMap;
    SymbolMap symbolMap;

    // Forward declaration
    // Helper: collect all members from a scope, flattening generate blocks
    auto collectMembers = [](auto& self, const slang::ast::Scope& scope,
                             std::vector<const slang::ast::Symbol*>& out) -> void {
        for (auto& member : scope.members()) {
            if (member.kind == slang::ast::SymbolKind::GenerateBlock) {
                auto& gb = member.as<slang::ast::GenerateBlockSymbol>();
                if (!gb.isUninstantiated)
                    self(self, gb, out);
            } else if (member.kind == slang::ast::SymbolKind::GenerateBlockArray) {
                auto& gba = member.as<slang::ast::GenerateBlockArraySymbol>();
                for (auto* entry : gba.entries) {
                    if (entry && !entry->isUninstantiated)
                        self(self, *entry, out);
                }
            } else {
                out.push_back(&member);
            }
        }
    };

    auto lowerInstance = [&](auto& self, const slang::ast::InstanceSymbol& inst,
                             const std::string& prefix, bool isTop) -> void {
        auto& body = inst.body;

        // Flatten all members including those inside generate blocks
        std::vector<const slang::ast::Symbol*> allMembers;
        collectMembers(collectMembers, body, allMembers);

        // ── Collect signals for this scope ──────────────────────────────────

        // Ports (these are always at the top level, not inside generate)
        for (auto& member : body.members()) {
            if (member.kind == slang::ast::SymbolKind::Port) {
                auto& port = member.as<slang::ast::PortSymbol>();
                if (isTop) {
                    // Top-level ports become I/O signals
                    SignalKind sk;
                    switch (port.direction) {
                        case slang::ast::ArgumentDirection::In:    sk = SignalKind::Input; break;
                        case slang::ast::ArgumentDirection::Out:   sk = SignalKind::Output; break;
                        case slang::ast::ArgumentDirection::InOut:  sk = SignalKind::Output; break;
                        default: sk = SignalKind::Internal; break;
                    }
                    uint32_t w = getTypeWidth(port.getType());
                    uint32_t idx = mod->addSignal(std::string(port.name), w, sk);
                    symbolMap[&port] = idx;
                    // Also map the internal symbol if it exists
                    if (port.internalSymbol)
                        symbolMap[port.internalSymbol] = idx;
                }
                // Child ports are mapped during port binding (below)
            }
        }

        // Variables and nets (including those inside generate blocks)
        for (auto* memberPtr : allMembers) {
            auto& member = *memberPtr;
            if (member.kind == slang::ast::SymbolKind::Variable) {
                auto& var = member.as<slang::ast::VariableSymbol>();
                // Skip if already mapped (e.g., port internal symbol)
                if (symbolMap.count(&var)) continue;
                auto arrInfo = getUnpackedArrayInfo(var.getType());
                if (arrInfo) {
                    auto [elemWidth, arrSize, rangeLower] = *arrInfo;
                    uint32_t baseIdx = static_cast<uint32_t>(mod->signals.size());
                    for (uint32_t i = 0; i < arrSize; i++) {
                        std::string elemName = prefix + std::string(var.name)
                            + "[" + std::to_string(i) + "]";
                        mod->addSignal(elemName, elemWidth, SignalKind::Internal);
                    }
                    arrayMap[&var] = ArrayInfo{baseIdx, arrSize, elemWidth, rangeLower};
                } else {
                    uint32_t w = getTypeWidth(var.getType());
                    uint32_t idx = mod->addSignal(prefix + std::string(var.name),
                                                   w, SignalKind::Internal);
                    symbolMap[&var] = idx;
                }
            }
            if (member.kind == slang::ast::SymbolKind::Net) {
                auto& net = member.as<slang::ast::NetSymbol>();
                if (symbolMap.count(&net)) continue;
                auto arrInfo = getUnpackedArrayInfo(net.getType());
                if (arrInfo) {
                    auto [elemWidth, arrSize, rangeLower] = *arrInfo;
                    uint32_t baseIdx = static_cast<uint32_t>(mod->signals.size());
                    for (uint32_t i = 0; i < arrSize; i++) {
                        std::string elemName = prefix + std::string(net.name)
                            + "[" + std::to_string(i) + "]";
                        mod->addSignal(elemName, elemWidth, SignalKind::Internal);
                    }
                    arrayMap[&net] = ArrayInfo{baseIdx, arrSize, elemWidth, rangeLower};
                } else {
                    uint32_t w = getTypeWidth(net.getType());
                    uint32_t idx = mod->addSignal(prefix + std::string(net.name),
                                                   w, SignalKind::Internal);
                    symbolMap[&net] = idx;
                }
            }
        }

        // ── Handle child instances (recursive inline) ───────────────────────

        for (auto* memberPtr : allMembers) {
            auto& member = *memberPtr;
            if (member.kind == slang::ast::SymbolKind::Instance) {
                auto& childInst = member.as<slang::ast::InstanceSymbol>();
                std::string childPrefix = prefix + std::string(childInst.name) + ".";

                // Helper: map a child port (and its internal symbols) to a signal index
                auto mapChildPort = [&](const slang::ast::Symbol& portSym,
                                        uint32_t sigIdx) {
                    symbolMap[&portSym] = sigIdx;
                    // Map the port's internal symbol
                    if (portSym.kind == slang::ast::SymbolKind::Port) {
                        auto& ps = portSym.as<slang::ast::PortSymbol>();
                        if (ps.internalSymbol)
                            symbolMap[ps.internalSymbol] = sigIdx;
                    }
                    // Also search the child body for any variable/net matching
                    // the port name (robust against internalSymbol mismatches)
                    auto& childBody = childInst.body;
                    for (auto& m : childBody.members()) {
                        if (m.kind == slang::ast::SymbolKind::Variable
                            && m.name == portSym.name) {
                            symbolMap[&m] = sigIdx;
                        }
                        if (m.kind == slang::ast::SymbolKind::Net
                            && m.name == portSym.name) {
                            symbolMap[&m] = sigIdx;
                        }
                    }
                };

                // Port binding: map child ports to parent signals
                auto connections = childInst.getPortConnections();
                for (const auto* conn : connections) {
                    auto& portSym = conn->port;
                    const auto* connExpr = conn->getExpression();

                    // Extract the parent signal symbol from the connection expression
                    const slang::ast::Symbol* parentSym = nullptr;
                    if (connExpr) {
                        if (connExpr->kind == slang::ast::ExpressionKind::NamedValue) {
                            parentSym = &connExpr->as<slang::ast::NamedValueExpression>().symbol;
                        } else if (connExpr->kind == slang::ast::ExpressionKind::Assignment) {
                            // Output port: slang creates assignment expr (parent_var = port)
                            auto& ae = connExpr->as<slang::ast::AssignmentExpression>();
                            if (ae.left().kind == slang::ast::ExpressionKind::NamedValue)
                                parentSym = &ae.left().as<slang::ast::NamedValueExpression>().symbol;
                        } else if (connExpr->kind == slang::ast::ExpressionKind::Conversion) {
                            auto& inner = connExpr->as<slang::ast::ConversionExpression>().operand();
                            if (inner.kind == slang::ast::ExpressionKind::NamedValue)
                                parentSym = &inner.as<slang::ast::NamedValueExpression>().symbol;
                        }
                    }

                    if (parentSym) {
                        // Simple named connection: child port maps to parent signal
                        auto parentIt = symbolMap.find(parentSym);
                        if (parentIt != symbolMap.end()) {
                            mapChildPort(portSym, parentIt->second);
                        } else {
                            auto* parentSig = mod->findSignal(std::string(parentSym->name));
                            if (parentSig) {
                                mapChildPort(portSym, parentSig->index);
                            } else {
                                std::cerr << "surge: cannot resolve port connection for '"
                                          << portSym.name << "'\n";
                            }
                        }
                    } else if (connExpr) {
                        // Complex expression connection: create an internal signal
                        uint32_t w = getTypeWidth(connExpr->type->getCanonicalType());
                        uint32_t idx = mod->addSignal(childPrefix + std::string(portSym.name),
                                                       w, SignalKind::Internal);
                        mapChildPort(portSym, idx);
                    } else {
                        // Unconnected port: create a dummy signal
                        uint32_t w = 1;
                        if (portSym.kind == slang::ast::SymbolKind::Port)
                            w = getTypeWidth(portSym.as<slang::ast::PortSymbol>().getType());
                        uint32_t idx = mod->addSignal(childPrefix + std::string(portSym.name),
                                                       w, SignalKind::Internal);
                        mapChildPort(portSym, idx);
                    }
                }

                // Recurse into child instance
                self(self, childInst, childPrefix, false);
            }
        }

        // ── Lower processes ─────────────────────────────────────────────────

        ExprLowering exprLower(*mod, arrayMap, symbolMap);

        for (auto* memberPtr : allMembers) {
            auto& member = *memberPtr;
            if (member.kind == slang::ast::SymbolKind::ProceduralBlock) {
                auto& pb = member.as<slang::ast::ProceduralBlockSymbol>();
                Process proc;

                if (pb.procedureKind == slang::ast::ProceduralBlockKind::AlwaysFF) {
                    proc.kind = Process::Sequential;
                    // Find the clock signal for this scope
                    auto* clkSig = mod->findSignal(prefix.empty() ? "clk" : prefix + "clk");
                    if (!clkSig) clkSig = mod->findSignal("clk"); // fallback to top-level clk
                    if (clkSig) {
                        proc.clockSignalIndex = clkSig->index;
                        proc.clockEdge = EdgeKind::Posedge;
                    }
                } else {
                    proc.kind = Process::Combinational;
                }

                StmtLowering stmtLower(*mod, exprLower, symbolMap);
                stmtLower.lower(pb.getBody());
                proc.assignments = std::move(stmtLower.assignments());

                if (proc.kind == Process::Sequential) {
                    for (auto& a : proc.assignments) {
                        if (a.indexExpr) {
                            for (uint32_t i = 0; i < a.arraySize; i++)
                                mod->signals[a.targetIndex + i].isFF = true;
                        } else {
                            mod->signals[a.targetIndex].isFF = true;
                        }
                    }
                }

                mod->processes.push_back(std::move(proc));
            }

            if (member.kind == slang::ast::SymbolKind::ContinuousAssign) {
                auto& ca = member.as<slang::ast::ContinuousAssignSymbol>();
                auto& assign = ca.getAssignment();

                Process proc;
                proc.kind = Process::Combinational;

                if (assign.kind == slang::ast::ExpressionKind::Assignment) {
                    auto& ae = assign.as<slang::ast::AssignmentExpression>();
                    auto& lhs = ae.left();
                    if (lhs.kind == slang::ast::ExpressionKind::NamedValue) {
                        auto& nv = lhs.as<slang::ast::NamedValueExpression>();
                        // Use symbolMap for LHS resolution
                        auto symIt = symbolMap.find(&nv.symbol);
                        uint32_t sigIdx = UINT32_MAX;
                        if (symIt != symbolMap.end()) {
                            sigIdx = symIt->second;
                        } else {
                            auto* sig = mod->findSignal(std::string(nv.symbol.name));
                            if (sig) sigIdx = sig->index;
                        }
                        if (sigIdx != UINT32_MAX) {
                            auto rhs = exprLower.lower(ae.right());
                            proc.assignments.push_back({sigIdx, rhs});
                        }
                    } else if (lhs.kind == slang::ast::ExpressionKind::Concatenation) {
                        auto& cc = lhs.as<slang::ast::ConcatenationExpression>();
                        auto rhs = exprLower.lower(ae.right());
                        std::vector<std::pair<const slang::ast::Expression*, uint32_t>> targets;
                        for (auto* op : cc.operands())
                            targets.push_back({op, getTypeWidth(*op->type)});
                        uint32_t bitPos = 0;
                        for (int i = static_cast<int>(targets.size()) - 1; i >= 0; i--) {
                            auto [tExpr, tWidth] = targets[i];
                            if (tExpr->kind == slang::ast::ExpressionKind::NamedValue) {
                                auto& nv = tExpr->as<slang::ast::NamedValueExpression>();
                                auto symIt = symbolMap.find(&nv.symbol);
                                uint32_t sigIdx = UINT32_MAX;
                                if (symIt != symbolMap.end()) sigIdx = symIt->second;
                                else {
                                    auto* sig = mod->findSignal(std::string(nv.symbol.name));
                                    if (sig) sigIdx = sig->index;
                                }
                                if (sigIdx != UINT32_MAX) {
                                    auto sliced = Expr::slice(tWidth, rhs,
                                        bitPos + tWidth - 1, bitPos);
                                    proc.assignments.push_back({sigIdx, sliced});
                                }
                            }
                            bitPos += tWidth;
                        }
                    } else if (lhs.kind == slang::ast::ExpressionKind::ElementSelect) {
                        auto& es = lhs.as<slang::ast::ElementSelectExpression>();
                        auto rhs = exprLower.lower(ae.right());
                        const slang::ast::Symbol* baseSym = nullptr;
                        if (es.value().kind == slang::ast::ExpressionKind::NamedValue)
                            baseSym = &es.value().as<slang::ast::NamedValueExpression>().symbol;
                        if (baseSym) {
                            auto arrIt = arrayMap.find(baseSym);
                            if (arrIt != arrayMap.end()) {
                                auto& info = arrIt->second;
                                auto constIdx = extractConstantInt(es.selector());
                                if (constIdx) {
                                    uint32_t idx = static_cast<uint32_t>(*constIdx)
                                        - static_cast<uint32_t>(info.rangeLower);
                                    if (idx < info.size) {
                                        proc.assignments.push_back(
                                            {info.baseSignalIndex + idx, rhs});
                                    }
                                }
                            }
                        }
                    }
                }

                if (!proc.assignments.empty())
                    mod->processes.push_back(std::move(proc));
            }
        }
    };

    // ── Recursively inline from the top instance ────────────────────────────

    lowerInstance(lowerInstance, *topInst, "", true);

    mod->computeLayout();

    std::cerr << "surge: built IR for module '" << mod->name
              << "' (" << mod->signals.size() << " signals, "
              << mod->processes.size() << " processes, "
              << mod->stateSize << " bytes state)\n";

    return mod;
}

} // namespace surge::ir
