import ctypes
import time
import numpy as np
from numpyler.runtime import numpy_to_memref, MemRefDescriptor
from numpyler.compiler.engine import initialize_llvm, create_execution_engine, compile_ir
from numpyler.utils.read_file import load_ir_from_file

# Global cache for expensive LLVM operations
_llvm_initialized = False
_engine_cache = {}
_ir_cache = {}

def get_cached_engine_and_function(operation="vector_mul"):
    """Get or create cached LLVM engine and compiled function"""
    global _llvm_initialized
    
    if operation in _engine_cache:
        return _engine_cache[operation]
    
    print(f"[CACHE] Creating new engine for {operation}")
    
    # Initialize LLVM only once globally
    if not _llvm_initialized:
        print("[CACHE] Initializing LLVM (one-time)")
        initialize_llvm()
        _llvm_initialized = True
    
    # Create engine
    engine = create_execution_engine()
    
    # Load and cache IR
    if operation not in _ir_cache:
        ir_file = "numpyler/llvm_ir/vector_mul.ll"  # Map operation to IR file
        print(f"[CACHE] Loading IR from {ir_file}")
        _ir_cache[operation] = load_ir_from_file(ir_file)
    
    # Compile IR
    print(f"[CACHE] Compiling IR for {operation}")
    mod = compile_ir(engine, _ir_cache[operation])
    
    # Get function pointer and create ctypes function
    func_ptr = engine.get_function_address("vector_mul")
    cfunc = ctypes.CFUNCTYPE(None,
                        ctypes.POINTER(MemRefDescriptor),
                        ctypes.POINTER(MemRefDescriptor),
                        ctypes.POINTER(MemRefDescriptor))(func_ptr)
    
    # Cache everything
    _engine_cache[operation] = (engine, mod, cfunc)
    print(f"[CACHE] Cached engine for {operation}")
    
    return _engine_cache[operation]

def fast_compile_and_run(traced_array_a, traced_array_b):
    engine, mod, cfunc = get_cached_engine_and_function()

    # Prepare input arrays (copies)
    arr_a = traced_array_a.data.copy() if hasattr(traced_array_a, 'data') else traced_array_a.realize()
    arr_b = traced_array_b.data.copy() if hasattr(traced_array_b, 'data') else traced_array_b.realize()

    # Create output array
    out = np.empty_like(arr_a)

    # Convert numpy arrays to memrefs
    a_memref = numpy_to_memref(arr_a)
    b_memref = numpy_to_memref(arr_b)
    out_memref = numpy_to_memref(out)

    # Call the LLVM compiled function
    cfunc(ctypes.byref(a_memref), ctypes.byref(b_memref), ctypes.byref(out_memref))

    return out

def compile_and_run(a_memref, b_memref, out_memref):
    # We can do the same as fast_compile_and_run but with memrefs directly
    engine, mod, cfunc = get_cached_engine_and_function()
    cfunc(ctypes.byref(a_memref), ctypes.byref(b_memref), ctypes.byref(out_memref))


# Utility to clear cache if needed
def clear_llvm_cache():
    """Clear all cached LLVM resources"""
    global _llvm_initialized, _engine_cache, _ir_cache
    _llvm_initialized = False
    _engine_cache.clear()
    _ir_cache.clear()
    print("[CACHE] Cleared all LLVM caches")