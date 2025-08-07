import numpy as np
from llvmlite import ir
from numpyler.tracing import TracedArray

def dtype_to_llvm(dtype):
    if dtype == np.int32:
        return ir.IntType(32)
    elif dtype == np.int64:
        return ir.IntType(64)
    elif dtype == np.float32:
        return ir.FloatType()
    elif dtype == np.float64:
        return ir.DoubleType()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def promote_inputs(builder, vals, input_types, output_type):
    """
    Promote input values to output_type to ensure consistent operation types.
    Handles integer to float conversion as needed.
    
    Args:
        builder: LLVM IRBuilder instance.
        vals: list of LLVM values (inputs).
        input_types: list of LLVM types of the input values.
        output_type: LLVM type of the output (target type).
        
    Returns:
        List of promoted LLVM values, all matching output_type.
    """
    promoted = []
    for val, in_ty in zip(vals, input_types):
        if in_ty == output_type:
            promoted.append(val)
        else:
            # Promote integer to float if output_type is float
            if isinstance(in_ty, ir.IntType) and isinstance(output_type, (ir.FloatType, ir.DoubleType)):
                promoted.append(builder.sitofp(val, output_type))
            # Demote float to int is not usual, but could add here if needed
            else:
                promoted.append(val)  # fallback, no promotion
    return promoted

def generate_fused_ir(nodes, leaf_arrays, output_dtype, output_shape, func_name, index_map):
    # Create LLVM module
    module = ir.Module(name=func_name)
    module.triple = "x86_64-pc-linux-gnu"
    module.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

    # Create memref type
    int64 = ir.IntType(64)
    int32 = ir.IntType(32)
    int8ptr = ir.IntType(8).as_pointer()
    
    # Create array types for shape and stride
    shape_array_type = ir.ArrayType(int64, 1)
    stride_array_type = ir.ArrayType(int64, 1)
    
    memref_type = ir.LiteralStructType([
        int8ptr,         # allocated
        int8ptr,         # aligned  
        int64,           # offset
        shape_array_type, # shape
        stride_array_type # stride
    ])

    # Create function type
    arg_types = [memref_type.as_pointer() for _ in range(len(leaf_arrays))]
    arg_types.append(memref_type.as_pointer())  # Add output argument
    ftype = ir.FunctionType(ir.VoidType(), arg_types)

    # Create function
    func = ir.Function(module, ftype, name=func_name)
    func_args = list(func.args)
    
    # Separate inputs and output
    input_args = func_args[:-1]
    output_arg = func_args[-1]

    # Create blocks
    entry_block = func.append_basic_block(name="entry")
    loop_block = func.append_basic_block(name="loop")
    done_block = func.append_basic_block(name="done")

    builder = ir.IRBuilder(entry_block)
    
    # Get output size
    out_shape_ptr = builder.gep(output_arg, [ir.Constant(int32, 0), ir.Constant(int32, 3), ir.Constant(int32, 0)])
    size = builder.load(out_shape_ptr)
    
    # Branch to loop
    builder.branch(loop_block)

    # Build loop block
    builder.position_at_end(loop_block)
    
    # Create phi node for loop index
    phi = builder.phi(int64)
    phi.add_incoming(ir.Constant(int64, 0), entry_block)
    
    # Load input values
    input_values = []
    input_types = []
    llvm_type = dtype_to_llvm(output_dtype)
    for i, arg in enumerate(input_args):
        actual_input_dtype = leaf_arrays[i].data.dtype
        input_llvm_type = dtype_to_llvm(actual_input_dtype)

        # Get aligned pointer
        aligned_ptr = builder.gep(arg, [ir.Constant(int32, 0), ir.Constant(int32, 1)])
        aligned = builder.load(aligned_ptr)
        
        # Cast to data type
        data_ptr = builder.bitcast(aligned, input_llvm_type.as_pointer())
        
        # Get element pointer
        elem_ptr = builder.gep(data_ptr, [phi])
        val = builder.load(elem_ptr)
        input_values.append(val)
        input_types.append(input_llvm_type)
    
    # Promote inputs to output type
    input_values = promote_inputs(builder, input_values, input_types, llvm_type)

    # Process operation nodes
    val_count = 0
    node_registers = {}
    
    for node in nodes:
        input_vals = []
        for inp in node.inputs:
            if isinstance(inp, TracedArray) and inp.trace_node:
                input_vals.append(node_registers[inp.trace_node.id])
            elif isinstance(inp, TracedArray) and inp.original_index is not None:
                pos = index_map[inp.original_index]
                input_vals.append(input_values[pos])
            elif isinstance(inp, (int, float)):
                # Promote constants to output LLVM type
                if isinstance(llvm_type, ir.IntType):
                    input_vals.append(ir.Constant(llvm_type, int(inp)))
                else:
                    input_vals.append(ir.Constant(llvm_type, float(inp)))
        
        # Promote operands to output type 
        # Note: constants already promoted above
        # For nodes inputs from registers, types should already match, but double check:
        # Could add promotion here if needed

        if node.op_name == "add":
            if isinstance(llvm_type, ir.IntType):
                res = builder.add(input_vals[0], input_vals[1], name=f"v{val_count}")
            else:
                res = builder.fadd(input_vals[0], input_vals[1], name=f"v{val_count}")
        elif node.op_name == "multiply":
            if isinstance(llvm_type, ir.IntType):
                res = builder.mul(input_vals[0], input_vals[1], name=f"v{val_count}")
            else:
                res = builder.fmul(input_vals[0], input_vals[1], name=f"v{val_count}")
        elif node.op_name == "subtract":
            if isinstance(llvm_type, ir.IntType):
                res = builder.sub(input_vals[0], input_vals[1], name=f"v{val_count}")
            else:
                res = builder.fsub(input_vals[0], input_vals[1], name=f"v{val_count}")
        elif node.op_name == "divide":
            # Always perform floating point division
            fp_type = ir.DoubleType() if output_dtype == np.float64 else ir.FloatType()
            def to_fp(val, val_type):
                if isinstance(val_type, ir.IntType):
                    return builder.sitofp(val, fp_type)
                return val
            val0 = to_fp(input_vals[0], llvm_type)
            val1 = to_fp(input_vals[1], llvm_type)
            res = builder.fdiv(val0, val1, name=f"v{val_count}")
        else:
            raise ValueError(f"Unsupported operation: {node.op_name}")
        
        node_registers[node.id] = res
        val_count += 1
    
    # Store result
    result = node_registers[nodes[-1].id]
    out_aligned_ptr = builder.gep(output_arg, [ir.Constant(int32, 0), ir.Constant(int32, 1)])
    out_aligned = builder.load(out_aligned_ptr)
    out_data = builder.bitcast(out_aligned, llvm_type.as_pointer())
    out_addr = builder.gep(out_data, [phi])
    builder.store(result, out_addr)
    
    # Loop increment
    i_next = builder.add(phi, ir.Constant(int64, 1), name="i_next")
    phi.add_incoming(i_next, builder.block)
    
    # Loop condition
    cond = builder.icmp_signed('<', i_next, size, name="cond")
    builder.cbranch(cond, loop_block, done_block)
    
    # Done block
    builder.position_at_end(done_block)
    builder.ret_void()
    
    return str(module)
