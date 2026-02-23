#include "codegen.h"
#include "../ir/ir.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>

namespace surge::codegen {

// ── Impl ────────────────────────────────────────────────────────────────────

struct CompiledModule::Impl {
    std::unique_ptr<llvm::orc::LLJIT> jit;
};

CompiledModule::~CompiledModule() = default;
CompiledModule::CompiledModule(CompiledModule&&) noexcept = default;
CompiledModule& CompiledModule::operator=(CompiledModule&&) noexcept = default;

// ── LLVM init (once) ────────────────────────────────────────────────────────

namespace {
struct LLVMInit {
    LLVMInit() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};
static LLVMInit llvmInit;
} // namespace

// ── IR generation ───────────────────────────────────────────────────────────

namespace {

class IRGen {
public:
    IRGen(const ir::Module& mod, llvm::LLVMContext& ctx, llvm::Module& llvmMod)
        : mod_(mod), ctx_(ctx), llvmMod_(llvmMod), builder_(ctx) {}

    void generate() {
        // Create function: void eval(i8* state, i8* next_state)
        auto* i8PtrTy = llvm::PointerType::getUnqual(ctx_);
        auto* voidTy  = llvm::Type::getVoidTy(ctx_);
        auto* fnTy    = llvm::FunctionType::get(voidTy, {i8PtrTy, i8PtrTy}, false);
        evalFn_ = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                          "eval", llvmMod_);
        auto* state     = evalFn_->getArg(0);
        auto* nextState = evalFn_->getArg(1);
        state->setName("state");
        nextState->setName("next_state");

        auto* entry = llvm::BasicBlock::Create(ctx_, "entry", evalFn_);
        builder_.SetInsertPoint(entry);

        // Generate code for each process
        for (auto& proc : mod_.processes) {
            for (auto& assign : proc.assignments) {
                auto* val = emitExpr(assign.value, state);
                auto& sig = mod_.signals[assign.targetIndex];

                // Sequential assigns write to next_state, combinational to state
                auto* basePtr = (proc.kind == ir::Process::Sequential) ? nextState : state;
                storeSignal(basePtr, sig, val);
            }
        }

        builder_.CreateRetVoid();
    }

private:
    const ir::Module& mod_;
    llvm::LLVMContext& ctx_;
    llvm::Module& llvmMod_;
    llvm::IRBuilder<> builder_;
    llvm::Function* evalFn_ = nullptr;

    /// Get LLVM integer type for a signal width
    llvm::IntegerType* intTy(uint32_t bits) {
        return llvm::IntegerType::get(ctx_, std::max(bits, 1u));
    }

    /// Load a signal value from state
    llvm::Value* loadSignal(llvm::Value* base, const ir::Signal& sig) {
        uint32_t bytes = ir::bytesForWidth(sig.width);
        auto* storeTy = intTy(bytes * 8);
        auto* ptr = builder_.CreateGEP(
            llvm::Type::getInt8Ty(ctx_), base,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), sig.stateOffset));
        llvm::Value* val = builder_.CreateLoad(storeTy, ptr, sig.name);
        // Truncate to actual width if storage is wider
        if (bytes * 8 > sig.width)
            val = builder_.CreateTrunc(val, intTy(sig.width));
        return val;
    }

    /// Store a value to a signal in state
    void storeSignal(llvm::Value* base, const ir::Signal& sig, llvm::Value* val) {
        uint32_t bytes = ir::bytesForWidth(sig.width);
        auto* storeTy = intTy(bytes * 8);
        auto* ptr = builder_.CreateGEP(
            llvm::Type::getInt8Ty(ctx_), base,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), sig.stateOffset));
        // Extend if needed
        if (val->getType()->getIntegerBitWidth() < bytes * 8)
            val = builder_.CreateZExt(val, storeTy);
        else if (val->getType()->getIntegerBitWidth() > bytes * 8)
            val = builder_.CreateTrunc(val, storeTy);
        builder_.CreateStore(val, ptr);
    }

    /// Emit LLVM IR for a Surge IR expression
    llvm::Value* emitExpr(const ir::ExprPtr& expr, llvm::Value* stateBase) {
        switch (expr->kind) {
            case ir::ExprKind::Constant:
                return llvm::ConstantInt::get(intTy(expr->width), expr->constVal);

            case ir::ExprKind::SignalRef: {
                auto& sig = mod_.signals[expr->signalIndex];
                return loadSignal(stateBase, sig);
            }

            case ir::ExprKind::UnaryOp: {
                auto* operand = emitExpr(expr->operands[0], stateBase);
                return emitUnary(expr->unaryOp, operand, expr->width);
            }

            case ir::ExprKind::BinaryOp: {
                auto* lhs = emitExpr(expr->operands[0], stateBase);
                auto* rhs = emitExpr(expr->operands[1], stateBase);
                return emitBinary(expr->binaryOp, lhs, rhs, expr->width);
            }

            case ir::ExprKind::Mux: {
                auto* cond = emitExpr(expr->operands[0], stateBase);
                auto* trueVal = emitExpr(expr->operands[1], stateBase);
                auto* falseVal = emitExpr(expr->operands[2], stateBase);
                // Ensure cond is i1
                if (cond->getType()->getIntegerBitWidth() > 1)
                    cond = builder_.CreateICmpNE(cond,
                        llvm::ConstantInt::get(cond->getType(), 0));
                // Match widths
                trueVal = matchWidth(trueVal, expr->width);
                falseVal = matchWidth(falseVal, expr->width);
                return builder_.CreateSelect(cond, trueVal, falseVal);
            }

            case ir::ExprKind::Slice: {
                auto* src = emitExpr(expr->operands[0], stateBase);
                uint32_t lo = expr->sliceLo;
                uint32_t w  = expr->width;
                if (lo > 0)
                    src = builder_.CreateLShr(src,
                        llvm::ConstantInt::get(src->getType(), lo));
                return builder_.CreateTrunc(src, intTy(w));
            }

            case ir::ExprKind::Concat: {
                // Concat: MSB-first in operands list
                // Shift each part to its position and OR together
                llvm::Value* result = llvm::ConstantInt::get(intTy(expr->width), 0);
                uint32_t shift = 0;
                // Process LSB-first (reverse order)
                for (int i = static_cast<int>(expr->operands.size()) - 1; i >= 0; i--) {
                    auto* part = emitExpr(expr->operands[i], stateBase);
                    part = builder_.CreateZExt(part, intTy(expr->width));
                    if (shift > 0)
                        part = builder_.CreateShl(part,
                            llvm::ConstantInt::get(intTy(expr->width), shift));
                    result = builder_.CreateOr(result, part);
                    shift += expr->operands[i]->width;
                }
                return result;
            }

            default:
                return llvm::ConstantInt::get(intTy(expr->width), 0);
        }
    }

    llvm::Value* matchWidth(llvm::Value* val, uint32_t targetWidth) {
        uint32_t curWidth = val->getType()->getIntegerBitWidth();
        if (curWidth < targetWidth)
            return builder_.CreateZExt(val, intTy(targetWidth));
        if (curWidth > targetWidth)
            return builder_.CreateTrunc(val, intTy(targetWidth));
        return val;
    }

    llvm::Value* emitUnary(ir::UnaryOp op, llvm::Value* a, uint32_t w) {
        switch (op) {
            case ir::UnaryOp::Not:
                return builder_.CreateNot(a);
            case ir::UnaryOp::Negate:
                return builder_.CreateNeg(a);
            case ir::UnaryOp::ReduceAnd:
            case ir::UnaryOp::ReduceOr:
            case ir::UnaryOp::ReduceXor: {
                // Reduction: collapse all bits
                auto allOnes = llvm::ConstantInt::getAllOnesValue(a->getType());
                auto zero = llvm::ConstantInt::get(a->getType(), 0);
                llvm::Value* result;
                if (op == ir::UnaryOp::ReduceAnd)
                    result = builder_.CreateICmpEQ(a, allOnes);
                else if (op == ir::UnaryOp::ReduceOr)
                    result = builder_.CreateICmpNE(a, zero);
                else {
                    // XOR reduce: use ctpop (popcount) & 1
                    auto* pop = builder_.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, a);
                    result = builder_.CreateTrunc(pop, llvm::Type::getInt1Ty(ctx_));
                }
                return builder_.CreateZExt(result, intTy(w));
            }
        }
        return a; // unreachable
    }

    llvm::Value* emitBinary(ir::BinaryOp op, llvm::Value* l, llvm::Value* r, uint32_t w) {
        // Match operand widths to max(l, r, w)
        uint32_t lw = l->getType()->getIntegerBitWidth();
        uint32_t rw = r->getType()->getIntegerBitWidth();
        uint32_t maxW = std::max({lw, rw, w});
        l = matchWidth(l, maxW);
        r = matchWidth(r, maxW);

        llvm::Value* result;
        switch (op) {
            case ir::BinaryOp::And: result = builder_.CreateAnd(l, r); break;
            case ir::BinaryOp::Or:  result = builder_.CreateOr(l, r); break;
            case ir::BinaryOp::Xor: result = builder_.CreateXor(l, r); break;
            case ir::BinaryOp::Add: result = builder_.CreateAdd(l, r); break;
            case ir::BinaryOp::Sub: result = builder_.CreateSub(l, r); break;
            case ir::BinaryOp::Mul: result = builder_.CreateMul(l, r); break;
            case ir::BinaryOp::Shl: result = builder_.CreateShl(l, r); break;
            case ir::BinaryOp::Shr: result = builder_.CreateLShr(l, r); break;
            case ir::BinaryOp::AShr: result = builder_.CreateAShr(l, r); break;
            case ir::BinaryOp::Eq:  result = builder_.CreateZExt(builder_.CreateICmpEQ(l, r), intTy(w)); break;
            case ir::BinaryOp::Neq: result = builder_.CreateZExt(builder_.CreateICmpNE(l, r), intTy(w)); break;
            case ir::BinaryOp::Lt:  result = builder_.CreateZExt(builder_.CreateICmpULT(l, r), intTy(w)); break;
            case ir::BinaryOp::Lte: result = builder_.CreateZExt(builder_.CreateICmpULE(l, r), intTy(w)); break;
            case ir::BinaryOp::Gt:  result = builder_.CreateZExt(builder_.CreateICmpUGT(l, r), intTy(w)); break;
            case ir::BinaryOp::Gte: result = builder_.CreateZExt(builder_.CreateICmpUGE(l, r), intTy(w)); break;
            default: result = l; break;
        }
        return matchWidth(result, w);
    }
};

} // namespace

// ── Compiler ────────────────────────────────────────────────────────────────

Compiler::Compiler() = default;
Compiler::~Compiler() = default;

CompiledModule Compiler::compile(const ir::Module& mod) {
    CompiledModule result;

    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto llvmMod = std::make_unique<llvm::Module>("surge_" + mod.name, *ctx);

    IRGen gen(mod, *ctx, *llvmMod);
    gen.generate();

    // Verify
    std::string verifyErr;
    llvm::raw_string_ostream verifyStream(verifyErr);
    if (llvm::verifyModule(*llvmMod, &verifyStream)) {
        std::cerr << "surge: LLVM IR verification failed:\n" << verifyErr << "\n";
        llvmMod->print(llvm::errs(), nullptr);
        return result;
    }

    // Optimize
    if (optLevel_ > 0) {
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;

        llvm::PassBuilder PB;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        auto optLevel = (optLevel_ >= 3) ? llvm::OptimizationLevel::O3
                      : (optLevel_ >= 2) ? llvm::OptimizationLevel::O2
                                         : llvm::OptimizationLevel::O1;
        auto MPM = PB.buildPerModuleDefaultPipeline(optLevel);
        MPM.run(*llvmMod, MAM);
    }

    // Dump IR if requested (after optimization)
    if (dumpIR_) {
        llvm::raw_string_ostream irStream(result.irDump_);
        llvmMod->print(irStream, nullptr);
        std::cerr << result.irDump_ << "\n";
    }

    // JIT compile
    auto jitExpected = llvm::orc::LLJITBuilder().create();
    if (!jitExpected) {
        std::cerr << "surge: failed to create JIT: "
                  << llvm::toString(jitExpected.takeError()) << "\n";
        return result;
    }

    auto jit = std::move(*jitExpected);
    auto tsm = llvm::orc::ThreadSafeModule(std::move(llvmMod), std::move(ctx));
    if (auto err = jit->addIRModule(std::move(tsm))) {
        std::cerr << "surge: failed to add IR module: "
                  << llvm::toString(std::move(err)) << "\n";
        return result;
    }

    auto sym = jit->lookup("eval");
    if (!sym) {
        std::cerr << "surge: failed to lookup eval: "
                  << llvm::toString(sym.takeError()) << "\n";
        return result;
    }

    result.evalFn_ = sym->toPtr<void(uint8_t*, uint8_t*)>();
    result.impl_ = std::make_unique<CompiledModule::Impl>();
    result.impl_->jit = std::move(jit);

    std::cerr << "surge: JIT compiled eval function for '" << mod.name << "'\n";
    return result;
}

} // namespace surge::codegen
