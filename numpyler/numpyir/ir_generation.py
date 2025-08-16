import numpy as np
from numpyler.numpyir.tracing import TracedArray
from llvmlite import ir
from llvmlite.binding import get_default_triple
from numpyler.numpyir.memref_utils import (
    array_descriptor_type, get_data_ptr, get_size, get_shape_dim, 
    get_stride_dim, dtype_to_llvm
)
from numpyler.numpyir.ops.matmul_ir import generate_matmul_ir

def detect_matmul_pattern(nodes):
    return len(nodes) == 1 and nodes[0].op_name == "dot"

def generate_fused_ir_multidim(nodes, leaf_arrays, output_dtype, output_shape, func_name, index_map):
    if detect_matmul_pattern(nodes):
        return generate_matmul_ir(nodes[0], leaf_arrays, output_dtype, output_shape, func_name, index_map)
    else:
        return generate_elementwise_ir(nodes, leaf_arrays, output_dtype, output_shape, func_name, index_map)

def generate_elementwise_ir(nodes, leaf_arrays, output_dtype, output_shape, func_name, index_map):
    module = ir.Module(name=func_name)
    module.triple = get_default_triple()
    
    int64 = ir.IntType(64)
    int32 = ir.IntType(32)
    
    array_desc_type = array_descriptor_type()
    
    arg_types = [array_desc_type.as_pointer() for _ in range(len(leaf_arrays) + 1)]
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
        data_ptr = get_data_ptr(builder, arg)
        typed_ptr = builder.bitcast(data_ptr, input_llvm_type.as_pointer())
        val = builder.load(builder.gep(typed_ptr, [phi]))
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
                else:
                    if isinstance(inp.data, (int, float, np.integer, np.floating)):
                        input_vals.append(ir.Constant(llvm_type, float(inp.data)))
                    elif inp.data.ndim == 0:
                        input_vals.append(ir.Constant(llvm_type, float(inp.data.item())))
            elif isinstance(inp, (int, float, np.integer, np.floating)):
                input_vals.append(ir.Constant(llvm_type, float(inp)))
        
        if len(input_vals) < 2 and node.op_name in ["add", "multiply", "subtract", "divide"]:
            print(f"ERROR: Node {node.op_name} has {len(input_vals)} inputs: {node.inputs}")
            for i, inp in enumerate(node.inputs):
                if isinstance(inp, TracedArray):
                    print(f"  Input {i}: TracedArray, trace_node={inp.trace_node is not None}, original_index={inp.original_index}, data={inp.data}")
                else:
                    print(f"  Input {i}: {type(inp)}, value={inp}")
            raise RuntimeError(f"Binary operation {node.op_name} requires exactly 2 inputs, got {len(input_vals)}")
        
        if node.op_name == "add":
            res = builder.fadd(input_vals[0], input_vals[1])
        elif node.op_name == "multiply":
            res = builder.fmul(input_vals[0], input_vals[1])
        elif node.op_name == "subtract":
            res = builder.fsub(input_vals[0], input_vals[1])
        elif node.op_name == "divide":
            res = builder.fdiv(input_vals[0], input_vals[1])
        else:
            raise NotImplementedError(f"Unsupported op: {node.op_name}")
        
        node_registers[node.id] = res
    
    out_data_ptr = get_data_ptr(builder, func.args[-1])
    out_typed_ptr = builder.bitcast(out_data_ptr, llvm_type.as_pointer())
    builder.store(node_registers[nodes[-1].id], builder.gep(out_typed_ptr, [phi]))
    
    i_next = builder.add(phi, ir.Constant(int64, 1))
    phi.add_incoming(i_next, builder.block)
    builder.cbranch(builder.icmp_signed('<', i_next, ir.Constant(int64, np.prod(output_shape))), loop, done)
    
    builder.position_at_end(done)
    builder.ret_void()
    return str(module)