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
    cfunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(MemRefDescriptor), ctypes.c_int)(func_ptr)
    
    # Cache everything
    _engine_cache[operation] = (engine, mod, cfunc)
    print(f"[CACHE] Cached engine for {operation}")
    
    return _engine_cache[operation]

def fast_compile_and_run(traced_array, scalar):
    """Optimized version that reuses cached LLVM resources"""
    # Get cached engine and function (expensive operations cached)
    engine, mod, cfunc = get_cached_engine_and_function()
    
    # Fast path: direct numpy array processing
    if hasattr(traced_array, 'data') and isinstance(traced_array.data, np.ndarray):
        arr = traced_array.data.copy()  # Work on copy
    else:
        arr = traced_array.realize()
    
    # Convert to memref (this is still expensive but unavoidable)
    memref = numpy_to_memref(arr)
    
    # Execute the compiled function
    cfunc(ctypes.byref(memref), scalar)
    
    return arr

def compile_and_run(traced_array, scalar):
    """Original function - now calls the fast version"""
    return fast_compile_and_run(traced_array, scalar)

# Utility to clear cache if needed
def clear_llvm_cache():
    """Clear all cached LLVM resources"""
    global _llvm_initialized, _engine_cache, _ir_cache
    _llvm_initialized = False
    _engine_cache.clear()
    _ir_cache.clear()
    print("[CACHE] Cleared all LLVM caches")