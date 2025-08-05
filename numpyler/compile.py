import time
from functools import wraps
from numpyler.tracing import TracedArray, dump_trace
from numpyler.compiler.runner import compile_and_run
import numpy as np

_compile_cache = {}

def compile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        total_start = time.perf_counter()
        
        # Create cache key from input types/shapes before tracing
        cache_key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                cache_key_parts.append(('array', arg.shape, arg.dtype))
            elif isinstance(arg, (int, float)):
                cache_key_parts.append(('scalar', type(arg)))
            else:
                cache_key_parts.append(('other', type(arg)))
        
        cache_key = (func.__name__, tuple(cache_key_parts))
        key_time = time.perf_counter()
        
        # If we have a cached compiled version, use it directly
        if cache_key in _compile_cache:
            cache_lookup_time = time.perf_counter() - key_time
            compiled_func = _compile_cache[cache_key]
            
            exec_start = time.perf_counter()
            result = compiled_func(*args, **kwargs)
            exec_time = time.perf_counter() - exec_start
            
            total_time = time.perf_counter() - total_start
            
            # Only print timing for first few calls to avoid spam
            if not hasattr(wrapper, '_call_count'):
                wrapper._call_count = 0
            wrapper._call_count += 1
            
            if wrapper._call_count <= 3:
                print(f"[TIMING] Cache lookup: {cache_lookup_time*1000:.3f}ms, "
                      f"Execution: {exec_time*1000:.3f}ms, "
                      f"Total: {total_time*1000:.3f}ms")
            
            return result
        
        # First time - trace the function to understand the operation
        trace_start = time.perf_counter()
        traced_args = [
            TracedArray(arg) if isinstance(arg, (np.ndarray, int, float)) else arg
            for arg in args
        ]
        result = func(*traced_args, **kwargs)
        dump_trace(result)
        trace_time = time.perf_counter() - trace_start

        if isinstance(result, TracedArray) and result.trace_node:
            print(f"[COMPILE] Compiling new function for key {cache_key}")
            print(f"[TIMING] Tracing took: {trace_time*1000:.3f}ms")
            
           # Extract the final operation
            final_node = result.trace_node
            op_name = final_node.op_name
            op_inputs = final_node.inputs

            # Unwrap arguments from the trace
            lhs = op_inputs[0]
            rhs = op_inputs[1]

            lhs_val = lhs.data if isinstance(lhs, TracedArray) else lhs
            rhs_val = rhs.data if isinstance(rhs, TracedArray) else rhs

            compile_start = time.perf_counter()
            
            # Create the compiled function
            def compiled_func(*runtime_args):
                func_start = time.perf_counter()
                
                # Find which argument is the array and which is the scalar
                array_arg = None
                scalar_arg = None
                
                for arg in runtime_args:
                    if isinstance(arg, np.ndarray):
                        array_arg = arg
                    elif isinstance(arg, (int, float)):
                        scalar_arg = arg
                
                if array_arg is not None and scalar_arg is not None:
                    # Time the actual compiled execution
                    traced_result = TracedArray(array_arg.copy())  # Copy to avoid modifying original
                    llvm_start = time.perf_counter()
                    result = compile_and_run(traced_result, scalar_arg)
                    llvm_time = time.perf_counter() - llvm_start
                    
                    func_time = time.perf_counter() - func_start
                    
                    if not hasattr(compiled_func, '_exec_count'):
                        compiled_func._exec_count = 0
                    compiled_func._exec_count += 1
                    
                    if compiled_func._exec_count <= 3:
                        print(f"[TIMING] Compiled func overhead: {(func_time-llvm_time)*1000:.3f}ms, "
                              f"LLVM execution: {llvm_time*1000:.3f}ms")
                    
                    return result
                else:
                    # Fallback to numpy
                    print(f"[WARNING] Falling back to NumPy")
                    return func(*runtime_args, **kwargs)
            
            compile_time = time.perf_counter() - compile_start
            print(f"[TIMING] Function creation took: {compile_time*1000:.3f}ms")
            
            _compile_cache[cache_key] = compiled_func
            
            exec_start = time.perf_counter()
            final_result = compiled_func(*args, **kwargs)
            exec_time = time.perf_counter() - exec_start
            
            total_time = time.perf_counter() - total_start
            print(f"[TIMING] First execution: {exec_time*1000:.3f}ms, "
                  f"Total first call: {total_time*1000:.3f}ms")
            
            return final_result

        raise RuntimeError("No trace found — was the input traced?")
    return wrapper