# numpyler/compile.py
from collections import OrderedDict
import time
import hashlib
import numpy as np
from functools import wraps
from numpyler.tracing import TracedArray, dump_trace, collect_nodes
from numpyler.compiler.runner import compile_and_run
from numpyler.compiler.ir_generation import generate_fused_ir

_compile_cache = {}

def is_constant(x):
    return isinstance(x, (int, float)) or (isinstance(x, TracedArray) and 
           (isinstance(x.data, (int, float)) or x.data.ndim == 0))

def get_constant_value(x):
    if isinstance(x, TracedArray):
        return x.data.item() if x.data.ndim == 0 else x.data
    return x

def compile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        total_start = time.perf_counter()
        
        # Create cache key
        cache_key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                cache_key_parts.append(('array', arg.shape, arg.dtype))
            elif isinstance(arg, (int, float)):
                cache_key_parts.append(('scalar', type(arg)))
            else:
                cache_key_parts.append(('other', type(arg)))
        
        cache_key = (func.__name__, tuple(cache_key_parts))
        
        # Try cache
        if cache_key in _compile_cache:
            compiled_func = _compile_cache[cache_key]
            result = compiled_func(*args, **kwargs)
            return result
        
        # Tracing
        trace_start = time.perf_counter()
        traced_args = [
            TracedArray(arg, original_index=i) if isinstance(arg, (np.ndarray, int, float)) else arg
            for i, arg in enumerate(args)
        ]
        result = func(*traced_args, **kwargs)
        
        if not isinstance(result, TracedArray) or not result.trace_node:
            return result.data if isinstance(result, TracedArray) else result
        
        # Collect computation graph
        nodes = collect_nodes(result)
        print("\n[TRACING] Computation Graph:")
        dump_trace(result)
        leaf_arrays = OrderedDict()
        
        # Collect leaf arrays and constants
        for node in nodes:
            for inp in node.inputs:
                if isinstance(inp, TracedArray):
                    if inp.trace_node is None and not is_constant(inp):
                        leaf_arrays[inp.original_index] = inp
        
        # Create mapping from original index to leaf position
        index_map = {}
        for idx, (orig_idx, leaf) in enumerate(leaf_arrays.items()):
            index_map[orig_idx] = idx
        
        # Ordered by original argument position
        leaf_arrays = list(leaf_arrays.values())
        
        # Generate fused IR
        output_dtype = result.data.dtype
        output_shape = result.data.shape
        
        # Create unique function name
        func_hash = hashlib.md5(repr(cache_key).encode()).hexdigest()[:8]
        func_name = f"fused_{func_hash}"
        
        ir_code = generate_fused_ir(nodes, leaf_arrays, output_dtype, output_shape, func_name, index_map)
        print("\n[TRACING] Generated IR:")
        print(ir_code)
        
        # Create compiled function
        def compiled_func(*runtime_args):
            # Prepare inputs in order
            input_arrays = []
            for leaf in leaf_arrays:
                idx = leaf.original_index
                input_arrays.append(runtime_args[idx])
            
            # Create output array
            out = np.empty(output_shape, dtype=output_dtype)
            
            # Convert to memrefs
            from numpyler.runtime import numpy_to_memref
            input_memrefs = [numpy_to_memref(arr) for arr in input_arrays]
            out_memref = numpy_to_memref(out)
            
            # Execute
            compile_and_run(input_memrefs, out_memref, ir_code=ir_code, func_name=func_name)
            return out
        
        # Cache and return
        _compile_cache[cache_key] = compiled_func
        return compiled_func(*args, **kwargs)
    
    return wrapper