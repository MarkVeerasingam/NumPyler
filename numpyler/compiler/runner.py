# numpyler/compiler/runner.py
import ctypes
import llvmlite.binding as llvm
from numpyler.runtime import MemRefDescriptor

_llvm_initialized = False
_engine_cache = {}
_module_cache = {}

def initialize_llvm():
    global _llvm_initialized
    if not _llvm_initialized:
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        _llvm_initialized = True

def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), target_machine)
    return engine

def compile_ir(engine, llvm_ir, func_name):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    _module_cache[func_name] = mod
    func_ptr = engine.get_function_address(func_name)
    return func_ptr

def get_cached_engine_and_function(ir_code, func_name, num_args):
    initialize_llvm()
    
    if func_name in _engine_cache:
        return _engine_cache[func_name]
    
    engine = create_execution_engine()
    func_ptr = compile_ir(engine, ir_code, func_name)
    
    # Create function type with correct number of arguments
    cfunc_type = ctypes.CFUNCTYPE(None, *([ctypes.POINTER(MemRefDescriptor)] * num_args))
    cfunc = cfunc_type(func_ptr)
    
    _engine_cache[func_name] = (engine, cfunc)
    return _engine_cache[func_name]

def compile_and_run(input_memrefs, out_memref, ir_code, func_name):
    # Total arguments = inputs + output
    num_args = len(input_memrefs) + 1
    engine, cfunc = get_cached_engine_and_function(ir_code, func_name, num_args)
    
    # Prepare arguments: inputs first, then output
    args = [ctypes.byref(memref) for memref in input_memrefs]
    args.append(ctypes.byref(out_memref))
    
    # Call the function
    cfunc(*args)