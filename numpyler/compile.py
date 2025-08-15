# numpyler/compile.py
from collections import OrderedDict
import hashlib
import numpy as np
from functools import wraps
from numpyler.numpyir.tracing import TracedArray, collect_nodes
from numpyler.compiler.runner import compile_and_run
from numpyler.runtime import numpy_to_memref
from numpyler.numpyir.ir_generation import generate_fused_ir_multidim  

_compile_cache = {}

def is_constant(x):
    return isinstance(x, (int, float)) or (isinstance(x, TracedArray) and 
           (isinstance(x.data, (int, float)) or x.data.ndim == 0))

def compile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        cache_key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                cache_key_parts.append(('array', arg.shape, arg.dtype))
            elif isinstance(arg, (int, float)):
                cache_key_parts.append(('scalar', type(arg)))
        
        cache_key = (func.__name__, tuple(cache_key_parts))
        
        # Try cache
        if cache_key in _compile_cache:
            return _compile_cache[cache_key](*args, **kwargs)
        
        # Tracing
        traced_args = [
            TracedArray(arg, original_index=i) if isinstance(arg, (np.ndarray, int, float)) else arg
            for i, arg in enumerate(args)
        ]
        result = func(*traced_args, **kwargs)
        
        if not isinstance(result, TracedArray) or not result.trace_node:
            return result.data if isinstance(result, TracedArray) else result
        
        # Collect computation graph
        nodes = collect_nodes(result)
        leaf_arrays = OrderedDict()
        
        for node in nodes:
            for inp in node.inputs:
                if isinstance(inp, TracedArray) and inp.trace_node is None and not is_constant(inp):
                    leaf_arrays[inp.original_index] = inp
        
        # Create mapping from original index to leaf position
        index_map = {orig_idx: idx for idx, orig_idx in enumerate(leaf_arrays)}
        
        # Generate fused IR
        func_hash = hashlib.md5(repr(cache_key).encode()).hexdigest()[:8]
        func_name = f"fused_{func_hash}"
        
        ir_code = generate_fused_ir_multidim(
            nodes, list(leaf_arrays.values()), 
            result.data.dtype, result.data.shape, 
            func_name, index_map
        )
        
        print(ir_code)
        # Create compiled function
        def compiled_func(*runtime_args):
            input_arrays = [runtime_args[leaf.original_index] for leaf in leaf_arrays.values()]
            out = np.empty(result.data.shape, dtype=result.data.dtype)
            
            compile_and_run(
                [numpy_to_memref(arr) for arr in input_arrays],
                numpy_to_memref(out),
                ir_code=ir_code,
                func_name=func_name
            )
            return out
        
        # Cache and return
        _compile_cache[cache_key] = compiled_func
        return compiled_func(*args, **kwargs)
    
    return wrapper