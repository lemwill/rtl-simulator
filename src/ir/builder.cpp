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
#include <slang/ast/expressions/SelectExpressions.h>
#include <slang/ast/statements/ConditionalStatements.h>
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
    return std::nullopt;
}

// ── Expression lowering ─────────────────────────────────────────────────────

class ExprLowering {
public:
    explicit ExprLowering(Module& mod) : mod_(mod) {}

    ExprPtr lower(const slang::ast::Expression& expr) {
        switch (expr.kind) {
            case slang::ast::ExpressionKind::IntegerLiteral:
                return lowerIntLiteral(expr.as<slang::ast::IntegerLiteral>());
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
            default:
                std::cerr << "surge: unsupported expression kind "
                          << static_cast<int>(expr.kind) << "\n";
                return Expr::constant(1, 0);
        }
    }

private:
    Module& mod_;

    ExprPtr lowerIntLiteral(const slang::ast::IntegerLiteral& lit) {
        auto val = lit.getValue();
        uint64_t v = 0;
        if (auto optVal = val.as<uint64_t>())
            v = *optVal;
        return Expr::constant(getTypeWidth(*lit.type), v);
    }

    ExprPtr lowerNamedValue(const slang::ast::NamedValueExpression& nv) {
        auto& sym = nv.symbol;

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
            case slang::ast::BinaryOperator::ArithmeticShiftRight: op = BinaryOp::AShr; break;
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
        return Expr::binary(op, w, lhs, rhs);
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
        return lower(cv.operand());
    }

    ExprPtr lowerConcat(const slang::ast::ConcatenationExpression& cc) {
        std::vector<ExprPtr> parts;
        for (auto* op : cc.operands())
            parts.push_back(lower(*op));
        return Expr::concat(getTypeWidth(*cc.type), std::move(parts));
    }

    ExprPtr lowerElementSelect(const slang::ast::ElementSelectExpression& es) {
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

    ExprPtr lowerRangeSelect(const slang::ast::RangeSelectExpression& rs) {
        auto src = lower(rs.value());
        uint32_t resultWidth = getTypeWidth(*rs.type);
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

        std::cerr << "surge: unsupported range select (dynamic bounds)\n";
        return Expr::constant(resultWidth, 0);
    }
};

// ── Statement lowering (builds assignments) ─────────────────────────────────

class StmtLowering {
public:
    StmtLowering(Module& mod, ExprLowering& el) : mod_(mod), el_(el) {}

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
    std::vector<Assignment> assignments_;

    void lowerExprStmt(const slang::ast::ExpressionStatement& es) {
        auto& expr = es.expr;
        if (expr.kind == slang::ast::ExpressionKind::Assignment) {
            auto& assign = expr.as<slang::ast::AssignmentExpression>();
            lowerAssignment(assign);
        }
    }

    void lowerAssignment(const slang::ast::AssignmentExpression& ae) {
        auto& lhs = ae.left();
        if (lhs.kind != slang::ast::ExpressionKind::NamedValue) {
            std::cerr << "surge: unsupported LHS in assignment\n";
            return;
        }
        auto& nv = lhs.as<slang::ast::NamedValueExpression>();
        auto* sig = mod_.findSignal(std::string(nv.symbol.name));
        if (!sig) {
            std::cerr << "surge: unknown target signal '" << nv.symbol.name << "'\n";
            return;
        }
        auto rhs = el_.lower(ae.right());
        assignments_.push_back({sig->index, rhs});
    }

    void lowerConditional(const slang::ast::ConditionalStatement& cs) {
        auto& conditions = cs.conditions;
        ExprPtr cond;
        if (!conditions.empty())
            cond = el_.lower(*conditions[0].expr);
        else
            cond = Expr::constant(1, 1);

        StmtLowering ifLower(mod_, el_);
        ifLower.lower(cs.ifTrue);
        auto ifAssigns = std::move(ifLower.assignments());

        std::vector<Assignment> elseAssigns;
        if (cs.ifFalse) {
            StmtLowering elseLower(mod_, el_);
            elseLower.lower(*cs.ifFalse);
            elseAssigns = std::move(elseLower.assignments());
        }

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

        // Collect default assignments
        std::unordered_map<uint32_t, ExprPtr> currentValues;
        if (cs.defaultCase) {
            StmtLowering defaultLower(mod_, el_);
            defaultLower.lower(*cs.defaultCase);
            for (auto& a : defaultLower.assignments())
                currentValues[a.targetIndex] = a.value;
        }

        // Process items in reverse for correct priority (first match wins)
        for (int i = static_cast<int>(cs.items.size()) - 1; i >= 0; i--) {
            auto& item = cs.items[i];

            StmtLowering itemLower(mod_, el_);
            itemLower.lower(*item.stmt);
            auto itemAssigns = std::move(itemLower.assignments());

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

            // Wrap each assignment in a mux
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
        }

        for (auto& [targetIdx, val] : currentValues)
            assignments_.push_back({targetIdx, val});
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
    auto& body = topInst->body;

    // ── Pass 1: collect signals ─────────────────────────────────────────────

    for (auto& member : body.members()) {
        if (member.kind == slang::ast::SymbolKind::Port) {
            auto& port = member.as<slang::ast::PortSymbol>();
            SignalKind sk;
            switch (port.direction) {
                case slang::ast::ArgumentDirection::In:    sk = SignalKind::Input; break;
                case slang::ast::ArgumentDirection::Out:   sk = SignalKind::Output; break;
                case slang::ast::ArgumentDirection::InOut:  sk = SignalKind::Output; break;
                default: sk = SignalKind::Internal; break;
            }
            uint32_t w = getTypeWidth(port.getType());
            mod->addSignal(std::string(port.name), w, sk);
        }
    }

    for (auto& member : body.members()) {
        if (member.kind == slang::ast::SymbolKind::Variable) {
            auto& var = member.as<slang::ast::VariableSymbol>();
            if (!mod->findSignal(std::string(var.name))) {
                uint32_t w = getTypeWidth(var.getType());
                mod->addSignal(std::string(var.name), w, SignalKind::Internal);
            }
        }
        if (member.kind == slang::ast::SymbolKind::Net) {
            auto& net = member.as<slang::ast::NetSymbol>();
            if (!mod->findSignal(std::string(net.name))) {
                uint32_t w = getTypeWidth(net.getType());
                mod->addSignal(std::string(net.name), w, SignalKind::Internal);
            }
        }
    }

    // ── Pass 2: lower processes ─────────────────────────────────────────────

    ExprLowering exprLower(*mod);

    for (auto& member : body.members()) {
        if (member.kind == slang::ast::SymbolKind::ProceduralBlock) {
            auto& pb = member.as<slang::ast::ProceduralBlockSymbol>();
            Process proc;

            if (pb.procedureKind == slang::ast::ProceduralBlockKind::AlwaysFF) {
                proc.kind = Process::Sequential;
                auto* clkSig = mod->findSignal("clk");
                if (clkSig) {
                    proc.clockSignalIndex = clkSig->index;
                    proc.clockEdge = EdgeKind::Posedge;
                }
            } else {
                proc.kind = Process::Combinational;
            }

            StmtLowering stmtLower(*mod, exprLower);
            stmtLower.lower(pb.getBody());
            proc.assignments = std::move(stmtLower.assignments());

            if (proc.kind == Process::Sequential) {
                for (auto& a : proc.assignments)
                    mod->signals[a.targetIndex].isFF = true;
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
                    auto* sig = mod->findSignal(std::string(nv.symbol.name));
                    if (sig) {
                        auto rhs = exprLower.lower(ae.right());
                        proc.assignments.push_back({sig->index, rhs});
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
                            auto* sig = mod->findSignal(std::string(nv.symbol.name));
                            if (sig) {
                                auto sliced = Expr::slice(tWidth, rhs,
                                    bitPos + tWidth - 1, bitPos);
                                proc.assignments.push_back({sig->index, sliced});
                            }
                        }
                        bitPos += tWidth;
                    }
                }
            }

            if (!proc.assignments.empty())
                mod->processes.push_back(std::move(proc));
        }
    }

    mod->computeLayout();

    std::cerr << "surge: built IR for module '" << mod->name
              << "' (" << mod->signals.size() << " signals, "
              << mod->processes.size() << " processes, "
              << mod->stateSize << " bytes state)\n";

    return mod;
}

} // namespace surge::ir
