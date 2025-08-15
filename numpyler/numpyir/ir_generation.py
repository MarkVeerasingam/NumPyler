# numpyler/compiler/ir_generation.py
import numpy as np
from numpyler.numpyir.tracing import TracedArray
from llvmlite import ir
from llvmlite.binding import get_default_triple

def dtype_to_llvm(dtype):
    if dtype == np.int32:
        return ir.IntType(32)
    elif dtype == np.int64:
        return ir.IntType(64)
    elif dtype == np.float32:
        return ir.FloatType()
    elif dtype == np.float64:
        return ir.DoubleType()
    raise ValueError(f"Unsupported dtype: {dtype}")

def generate_fused_ir_multidim(nodes, leaf_arrays, output_dtype, output_shape, func_name, index_map):
    module = ir.Module(name=func_name)
    module.triple = get_default_triple()
    
    int64 = ir.IntType(64)
    int32 = ir.IntType(32)
    int8ptr = ir.IntType(8).as_pointer()
    
    memref_type = ir.LiteralStructType([
        int8ptr, int8ptr, int64, int64,
        ir.ArrayType(int64, 8), ir.ArrayType(int64, 8)
    ])
    
    arg_types = [memref_type.as_pointer() for _ in range(len(leaf_arrays) + 1)]
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), arg_types), name=func_name)
    
    entry = func.append_basic_block("entry")
    loop = func.append_basic_block("loop")
    done = func.append_basic_block("done")
    
    builder = ir.IRBuilder(entry)
    builder.branch(loop)
    
    builder.position_at_end(loop)
    phi = builder.phi(int64)
    phi.add_incoming(ir.Constant(int64, 0), entry)
    
    llvm_type = dtype_to_llvm(output_dtype)
    input_values = []
    
    for i, arg in enumerate(func.args[:-1]):
        input_dtype = leaf_arrays[i].data.dtype
        input_llvm_type = dtype_to_llvm(input_dtype)
        
        aligned = builder.load(builder.gep(arg, [ir.Constant(int32, 0), ir.Constant(int32, 1)]))
        val = builder.load(builder.gep(builder.bitcast(aligned, input_llvm_type.as_pointer()), [phi]))
        input_values.append(val)
    
    node_registers = {}
    for node in nodes:
        input_vals = []
        for inp in node.inputs:
            if isinstance(inp, TracedArray):
                if inp.trace_node:
                    input_vals.append(node_registers[inp.trace_node.id])
                elif inp.original_index is not None:
                    input_vals.append(input_values[index_map[inp.original_index]])
            elif isinstance(inp, (int, float)):
                input_vals.append(ir.Constant(llvm_type, inp))
        
        if node.op_name == "add":
            res = builder.fadd(input_vals[0], input_vals[1])
        elif node.op_name == "multiply":
            res = builder.fmul(input_vals[0], input_vals[1])
        elif node.op_name == "subtract":
            res = builder.fsub(input_vals[0], input_vals[1])
        elif node.op_name == "divide":
            res = builder.fdiv(input_vals[0], input_vals[1])
            
        node_registers[node.id] = res
        
    out_aligned = builder.load(builder.gep(func.args[-1], [ir.Constant(int32, 0), ir.Constant(int32, 1)]))
    builder.store(node_registers[nodes[-1].id], builder.gep(builder.bitcast(out_aligned, llvm_type.as_pointer()), [phi]))
    
    i_next = builder.add(phi, ir.Constant(int64, 1))
    phi.add_incoming(i_next, builder.block)
    builder.cbranch(builder.icmp_signed('<', i_next, ir.Constant(int64, np.prod(output_shape))), loop, done)
    
    builder.position_at_end(done)
    builder.ret_void()
    
    return str(module)