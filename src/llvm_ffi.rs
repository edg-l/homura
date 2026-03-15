use llvm_sys::prelude::{LLVMContextRef, LLVMModuleRef};

unsafe extern "C" {
    /// Translate an MLIR module (given as its top-level operation) to an LLVM
    /// IR module owned by the provided LLVM context.
    ///
    /// Returns null on failure. Caller is responsible for disposing the returned
    /// `LLVMModuleRef` with `LLVMDisposeModule` and the context with
    /// `LLVMContextDispose`.
    pub fn mlirTranslateModuleToLLVMIR(
        module: mlir_sys::MlirOperation,
        context: LLVMContextRef,
    ) -> LLVMModuleRef;
}

#[cfg(test)]
mod tests {
    #[test]
    fn llvm_target_machine_init() {
        use llvm_sys::core::LLVMDisposeMessage;
        use llvm_sys::target::*;
        use llvm_sys::target_machine::*;
        use std::ffi::CStr;

        unsafe {
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmPrinters();

            let triple = LLVMGetDefaultTargetTriple();
            assert!(!triple.is_null());
            let triple_str = CStr::from_ptr(triple).to_string_lossy();
            assert!(!triple_str.is_empty(), "target triple should be non-empty");

            let mut target = std::ptr::null_mut();
            let mut error_msg = std::ptr::null_mut();
            let res = LLVMGetTargetFromTriple(triple, &mut target, &mut error_msg);
            assert_eq!(res, 0, "LLVMGetTargetFromTriple should succeed");
            assert!(!target.is_null(), "target should be non-null");

            let cpu = LLVMGetHostCPUName();
            let features = LLVMGetHostCPUFeatures();
            let machine = LLVMCreateTargetMachine(
                target,
                triple,
                cpu,
                features,
                LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                LLVMRelocMode::LLVMRelocPIC,
                LLVMCodeModel::LLVMCodeModelDefault,
            );
            assert!(!machine.is_null(), "target machine should be non-null");

            LLVMDisposeTargetMachine(machine);
            LLVMDisposeMessage(triple);
            LLVMDisposeMessage(cpu);
            LLVMDisposeMessage(features);
        }
    }
}
