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

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>

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
        // Compile-time analysis (done once, reused by both functions)
        analyzeProcesses();
        sortedCombAssigns_ = topoSortCombAssigns();
        ffRegions_ = computeFFRegions();

        // Generate single-cycle eval function (used for VCD tracing mode)
        generateEval();

        // Generate JIT loop function (fast path: eval + commitFFs in a loop)
        generateSimulate();
    }

private:
    const ir::Module& mod_;
    llvm::LLVMContext& ctx_;
    llvm::Module& llvmMod_;
    llvm::IRBuilder<> builder_;

    // SSA signal promotion (compile-time analysis, reused by both functions)
    std::vector<bool> combWritten_;
    struct CombAssign { uint32_t procIdx, assignIdx, targetSig; };
    std::vector<CombAssign> sortedCombAssigns_;
    struct FFRegion { uint32_t offset, bytes; };
    std::vector<FFRegion> ffRegions_;

    // Per-function SSA values (reset for each generated function)
    std::vector<llvm::Value*> signalValues_;

    // ── Analysis (compile-time, done once) ──────────────────────────────

    void analyzeProcesses() {
        combWritten_.assign(mod_.signals.size(), false);
        for (auto& proc : mod_.processes) {
            if (proc.kind != ir::Process::Combinational) continue;
            for (auto& assign : proc.assignments)
                if (!assign.indexExpr)
                    combWritten_[assign.targetIndex] = true;
        }
    }

    void collectSignalRefs(const ir::ExprPtr& expr, std::vector<uint32_t>& refs) {
        if (!expr) return;
        if (expr->kind == ir::ExprKind::SignalRef)
            refs.push_back(expr->signalIndex);
        for (auto& op : expr->operands)
            collectSignalRefs(op, refs);
    }

    std::vector<CombAssign> topoSortCombAssigns() {
        std::vector<CombAssign> assigns;
        for (uint32_t p = 0; p < mod_.processes.size(); p++) {
            auto& proc = mod_.processes[p];
            if (proc.kind != ir::Process::Combinational) continue;
            for (uint32_t a = 0; a < proc.assignments.size(); a++) {
                if (proc.assignments[a].indexExpr) continue;
                assigns.push_back({p, a, proc.assignments[a].targetIndex});
            }
        }
        if (assigns.empty()) return assigns;

        std::unordered_map<uint32_t, uint32_t> sigToIdx;
        for (uint32_t i = 0; i < assigns.size(); i++)
            sigToIdx[assigns[i].targetSig] = i;

        uint32_t n = assigns.size();
        std::vector<std::vector<uint32_t>> adj(n);
        std::vector<uint32_t> inDeg(n, 0);

        for (uint32_t i = 0; i < n; i++) {
            auto& assign = mod_.processes[assigns[i].procIdx].assignments[assigns[i].assignIdx];
            std::vector<uint32_t> refs;
            collectSignalRefs(assign.value, refs);
            std::sort(refs.begin(), refs.end());
            refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
            for (auto ref : refs) {
                auto it = sigToIdx.find(ref);
                if (it != sigToIdx.end() && it->second != i) {
                    adj[it->second].push_back(i);
                    inDeg[i]++;
                }
            }
        }

        std::queue<uint32_t> q;
        for (uint32_t i = 0; i < n; i++)
            if (inDeg[i] == 0) q.push(i);

        std::vector<CombAssign> sorted;
        sorted.reserve(n);
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            sorted.push_back(assigns[u]);
            for (auto v : adj[u])
                if (--inDeg[v] == 0) q.push(v);
        }
        return (sorted.size() < n) ? assigns : sorted;
    }

    std::vector<FFRegion> computeFFRegions() {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (auto& sig : mod_.signals) {
            if (!sig.isFF) continue;
            ranges.push_back({sig.stateOffset, ir::bytesForWidth(sig.width)});
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<FFRegion> regions;
        for (auto& [off, len] : ranges) {
            if (!regions.empty() && regions.back().offset + regions.back().bytes == off)
                regions.back().bytes += len;
            else
                regions.push_back({off, len});
        }
        return regions;
    }

    // ── Code generation ─────────────────────────────────────────────────

    /// Emit the eval body (SSA signal promotion) at the current insert point.
    /// Reads from state, writes comb to state, writes seq to nextState.
    void emitEvalBody(llvm::Value* state, llvm::Value* nextState) {
        const uint32_t numSigs = mod_.signals.size();
        signalValues_.assign(numSigs, nullptr);

        // Pre-load all non-comb-written signals from state
        for (uint32_t i = 0; i < numSigs; i++) {
            if (!combWritten_[i])
                signalValues_[i] = loadSignal(state, mod_.signals[i]);
        }

        // Emit comb scalar assignments in topological order
        for (auto& [procIdx, assignIdx, targetSig] : sortedCombAssigns_) {
            auto& assign = mod_.processes[procIdx].assignments[assignIdx];
            auto* val = emitExpr(assign.value, state);
            signalValues_[targetSig] = matchWidth(val, mod_.signals[targetSig].width);
        }

        // Emit comb array stores (still through memory)
        for (auto& proc : mod_.processes) {
            if (proc.kind != ir::Process::Combinational) continue;
            for (auto& assign : proc.assignments) {
                if (!assign.indexExpr) continue;
                auto* val = emitExpr(assign.value, state);
                emitArrayStore(state, state, assign, val);
            }
        }

        // Store comb scalar values to state (for runtime/VCD to read)
        for (uint32_t i = 0; i < numSigs; i++) {
            if (combWritten_[i] && signalValues_[i])
                storeSignal(state, mod_.signals[i], signalValues_[i]);
        }

        // Emit sequential assignments (read SSA values, write to nextState)
        for (auto& proc : mod_.processes) {
            if (proc.kind != ir::Process::Sequential) continue;
            for (auto& assign : proc.assignments) {
                auto* val = emitExpr(assign.value, state);
                if (assign.indexExpr)
                    emitArrayStore(nextState, state, assign, val);
                else
                    storeSignal(nextState, mod_.signals[assign.targetIndex], val);
            }
        }
    }

    /// Emit inline commitFFs: memcpy from nextState to state for each FF region
    void emitCommitFFs(llvm::Value* state, llvm::Value* nextState) {
        auto* i8Ty  = llvm::Type::getInt8Ty(ctx_);
        auto* i32Ty = llvm::Type::getInt32Ty(ctx_);
        for (auto& r : ffRegions_) {
            auto* src = builder_.CreateGEP(i8Ty, nextState,
                llvm::ConstantInt::get(i32Ty, r.offset));
            auto* dst = builder_.CreateGEP(i8Ty, state,
                llvm::ConstantInt::get(i32Ty, r.offset));
            builder_.CreateMemCpy(dst, llvm::MaybeAlign(1), src,
                                  llvm::MaybeAlign(1), r.bytes);
        }
    }

    /// Generate: void eval(i8* state, i8* next_state)
    void generateEval() {
        auto* ptrTy  = llvm::PointerType::getUnqual(ctx_);
        auto* voidTy = llvm::Type::getVoidTy(ctx_);
        auto* fnTy   = llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, false);
        auto* fn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                           "eval", llvmMod_);
        fn->getArg(0)->setName("state");
        fn->getArg(1)->setName("next_state");
        fn->getArg(0)->addAttr(llvm::Attribute::NoAlias);
        fn->getArg(1)->addAttr(llvm::Attribute::NoAlias);

        auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn);
        builder_.SetInsertPoint(entry);
        emitEvalBody(fn->getArg(0), fn->getArg(1));
        builder_.CreateRetVoid();
    }

    /// Generate: void simulate(i8* state, i8* next_state, i64 cycles)
    /// Contains the simulation loop with inline commitFFs.
    void generateSimulate() {
        auto* ptrTy  = llvm::PointerType::getUnqual(ctx_);
        auto* voidTy = llvm::Type::getVoidTy(ctx_);
        auto* i64Ty  = llvm::Type::getInt64Ty(ctx_);
        auto* fnTy   = llvm::FunctionType::get(voidTy, {ptrTy, ptrTy, i64Ty}, false);
        auto* fn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                           "simulate", llvmMod_);
        fn->getArg(0)->setName("state");
        fn->getArg(1)->setName("next_state");
        fn->getArg(2)->setName("cycles");
        fn->getArg(0)->addAttr(llvm::Attribute::NoAlias);
        fn->getArg(1)->addAttr(llvm::Attribute::NoAlias);
        fn->addFnAttr(llvm::Attribute::NoUnwind);

        auto* state     = fn->getArg(0);
        auto* nextState = fn->getArg(1);
        auto* cycles    = fn->getArg(2);

        auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn);
        auto* loop  = llvm::BasicBlock::Create(ctx_, "loop", fn);
        auto* exit  = llvm::BasicBlock::Create(ctx_, "exit", fn);

        // Entry: branch to loop
        builder_.SetInsertPoint(entry);
        auto* zero = llvm::ConstantInt::get(i64Ty, 0);
        auto* skipLoop = builder_.CreateICmpEQ(cycles, zero);
        builder_.CreateCondBr(skipLoop, exit, loop);

        // Loop body: eval + commitFFs + increment
        builder_.SetInsertPoint(loop);
        auto* i = builder_.CreatePHI(i64Ty, 2, "i");
        i->addIncoming(zero, entry);

        emitEvalBody(state, nextState);
        emitCommitFFs(state, nextState);

        auto* iNext = builder_.CreateAdd(i, llvm::ConstantInt::get(i64Ty, 1));
        i->addIncoming(iNext, builder_.GetInsertBlock());
        auto* done = builder_.CreateICmpUGE(iNext, cycles);
        builder_.CreateCondBr(done, exit, loop);

        // Exit
        builder_.SetInsertPoint(exit);
        builder_.CreateRetVoid();
    }

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

    /// Computed array store: GEP with runtime index
    void emitArrayStore(llvm::Value* storeBase, llvm::Value* readBase,
                        const ir::Assignment& assign, llvm::Value* val) {
        auto* indexVal = emitExpr(assign.indexExpr, readBase);
        uint32_t baseIdx    = assign.targetIndex;
        uint32_t arrSize    = assign.arraySize;
        uint32_t elemWidth  = assign.elementWidth;
        uint32_t elemBytes  = ir::bytesForWidth(elemWidth);

        // Clamp index to [0, arrSize-1]
        auto* i32Ty = llvm::Type::getInt32Ty(ctx_);
        auto* idx32 = builder_.CreateZExtOrTrunc(indexVal, i32Ty);
        auto* maxIdx = llvm::ConstantInt::get(i32Ty, arrSize - 1);
        auto* oob = builder_.CreateICmpUGT(idx32, maxIdx);
        auto* clampedIdx = builder_.CreateSelect(oob, maxIdx, idx32);

        // Compute byte offset: baseOffset + clampedIdx * elemBytes
        uint32_t baseOffset = mod_.signals[baseIdx].stateOffset;
        auto* baseConst = llvm::ConstantInt::get(i32Ty, baseOffset);
        auto* elemBytesConst = llvm::ConstantInt::get(i32Ty, elemBytes);
        auto* offsetFromBase = builder_.CreateMul(clampedIdx, elemBytesConst);
        auto* totalOffset = builder_.CreateAdd(baseConst, offsetFromBase);

        // GEP into store base with computed offset
        auto* ptr = builder_.CreateGEP(
            llvm::Type::getInt8Ty(ctx_), storeBase, totalOffset);

        // Extend/truncate value to storage width and store
        auto* storeTy = intTy(elemBytes * 8);
        if (val->getType()->getIntegerBitWidth() < elemBytes * 8)
            val = builder_.CreateZExt(val, storeTy);
        else if (val->getType()->getIntegerBitWidth() > elemBytes * 8)
            val = builder_.CreateTrunc(val, storeTy);
        builder_.CreateStore(val, ptr);
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
                auto idx = expr->signalIndex;
                if (idx < signalValues_.size() && signalValues_[idx])
                    return signalValues_[idx];
                auto& sig = mod_.signals[idx];
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

            case ir::ExprKind::ArrayElement: {
                auto* indexVal = emitExpr(expr->operands[0], stateBase);
                uint32_t baseIdx = expr->arrayBaseIndex;
                uint32_t arrSize = expr->arraySize;
                uint32_t elemWidth = expr->elementWidth;
                uint32_t elemBytes = ir::bytesForWidth(elemWidth);

                // Clamp index to [0, arrSize-1]
                auto* i32Ty = llvm::Type::getInt32Ty(ctx_);
                auto* idx32 = builder_.CreateZExtOrTrunc(indexVal, i32Ty);
                auto* maxIdx = llvm::ConstantInt::get(i32Ty, arrSize - 1);
                auto* oob = builder_.CreateICmpUGT(idx32, maxIdx);
                auto* clampedIdx = builder_.CreateSelect(oob, maxIdx, idx32);

                // Compute byte offset: baseOffset + clampedIdx * elemBytes
                uint32_t baseOffset = mod_.signals[baseIdx].stateOffset;
                auto* baseConst = llvm::ConstantInt::get(i32Ty, baseOffset);
                auto* elemBytesConst = llvm::ConstantInt::get(i32Ty, elemBytes);
                auto* offsetFromBase = builder_.CreateMul(clampedIdx, elemBytesConst);
                auto* totalOffset = builder_.CreateAdd(baseConst, offsetFromBase);

                // GEP into state with computed offset
                auto* ptr = builder_.CreateGEP(
                    llvm::Type::getInt8Ty(ctx_), stateBase, totalOffset);

                // Load and truncate
                auto* storeTy = intTy(elemBytes * 8);
                llvm::Value* val = builder_.CreateLoad(storeTy, ptr, "arr_elem");
                if (elemBytes * 8 > elemWidth)
                    val = builder_.CreateTrunc(val, intTy(elemWidth));
                return val;
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

    auto evalSym = jit->lookup("eval");
    if (!evalSym) {
        std::cerr << "surge: failed to lookup eval: "
                  << llvm::toString(evalSym.takeError()) << "\n";
        return result;
    }
    result.evalFn_ = evalSym->toPtr<void(uint8_t*, uint8_t*)>();

    auto simSym = jit->lookup("simulate");
    if (!simSym) {
        std::cerr << "surge: failed to lookup simulate: "
                  << llvm::toString(simSym.takeError()) << "\n";
        return result;
    }
    result.simulateFn_ = simSym->toPtr<void(uint8_t*, uint8_t*, uint64_t)>();

    result.impl_ = std::make_unique<CompiledModule::Impl>();
    result.impl_->jit = std::move(jit);

    std::cerr << "surge: JIT compiled eval function for '" << mod.name << "'\n";
    return result;
}

} // namespace surge::codegen
