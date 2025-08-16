# numpyler/numpyir/ops/matmul_ir.py
from llvmlite import ir
from llvmlite.binding import get_default_triple
from numpyler.numpyir.memref_utils import (
    array_descriptor_type, get_data_ptr, get_shape_dim, 
    get_stride_dim, dtype_to_llvm
)

def generate_matmul_inline(builder, func, A_arg, B_arg, C_arg, elem_type):
    """Generate inline matrix multiplication IR within an existing function"""
    i32 = ir.IntType(32)
    i64 = ir.IntType(64)
    
    # Get data pointers and cast to element type
    A_data = get_data_ptr(builder, A_arg)
    B_data = get_data_ptr(builder, B_arg)
    C_data = get_data_ptr(builder, C_arg)
    
    A_ptr = builder.bitcast(A_data, elem_type.as_pointer())
    B_ptr = builder.bitcast(B_data, elem_type.as_pointer())
    C_ptr = builder.bitcast(C_data, elem_type.as_pointer())
    
    # Get dimensions
    M = get_shape_dim(builder, A_arg, 0)  # A.shape[0]
    K = get_shape_dim(builder, A_arg, 1)  # A.shape[1]
    N = get_shape_dim(builder, B_arg, 1)  # B.shape[1]
    
    # Get strides
    As0 = get_stride_dim(builder, A_arg, 0)
    As1 = get_stride_dim(builder, A_arg, 1)
    Bs0 = get_stride_dim(builder, B_arg, 0)
    Bs1 = get_stride_dim(builder, B_arg, 1)
    Cs0 = get_stride_dim(builder, C_arg, 0)
    Cs1 = get_stride_dim(builder, C_arg, 1)
    
    # Create loop blocks
    loop_i = func.append_basic_block("matmul_loop_i")
    body_i = func.append_basic_block("matmul_body_i")
    loop_j = func.append_basic_block("matmul_loop_j")
    body_j = func.append_basic_block("matmul_body_j")
    loop_k = func.append_basic_block("matmul_loop_k")
    body_k = func.append_basic_block("matmul_body_k")
    after_k = func.append_basic_block("matmul_after_k")
    after_j = func.append_basic_block("matmul_after_j")
    after_i = func.append_basic_block("matmul_after_i")
    
    # Initialize i loop
    i_phi = builder.alloca(i64, name="i")
    builder.store(ir.Constant(i64, 0), i_phi)
    builder.branch(loop_i)
    
    # i loop condition
    builder.position_at_end(loop_i)
    i_val = builder.load(i_phi)
    i_cond = builder.icmp_signed("<", i_val, M)
    builder.cbranch(i_cond, body_i, after_i)
    
    # i loop body - initialize j loop
    builder.position_at_end(body_i)
    j_phi = builder.alloca(i64, name="j")
    builder.store(ir.Constant(i64, 0), j_phi)
    builder.branch(loop_j)
    
    # j loop condition
    builder.position_at_end(loop_j)
    j_val = builder.load(j_phi)
    j_cond = builder.icmp_signed("<", j_val, N)
    builder.cbranch(j_cond, body_j, after_j)
    
    # j loop body - initialize sum and k loop
    builder.position_at_end(body_j)
    sum_slot = builder.alloca(elem_type, name="sum")
    builder.store(ir.Constant(elem_type, 0.0), sum_slot)
    
    k_phi = builder.alloca(i64, name="k")
    builder.store(ir.Constant(i64, 0), k_phi)
    builder.branch(loop_k)
    
    # k loop condition
    builder.position_at_end(loop_k)
    k_val = builder.load(k_phi)
    k_cond = builder.icmp_signed("<", k_val, K)
    builder.cbranch(k_cond, body_k, after_k)
    
    # k loop body - multiply and accumulate
    builder.position_at_end(body_k)
    # A[i,k] = A_ptr[i*As0 + k*As1]
    i_As0 = builder.mul(i_val, As0)
    k_As1 = builder.mul(k_val, As1)
    a_index = builder.add(i_As0, k_As1)
    a_ptr = builder.gep(A_ptr, [a_index])
    a_val = builder.load(a_ptr)
    
    # B[k,j] = B_ptr[k*Bs0 + j*Bs1]
    k_Bs0 = builder.mul(k_val, Bs0)
    j_Bs1 = builder.mul(j_val, Bs1)
    b_index = builder.add(k_Bs0, j_Bs1)
    b_ptr = builder.gep(B_ptr, [b_index])
    b_val = builder.load(b_ptr)
    
    # sum += A[i,k] * B[k,j]
    prod = builder.fmul(a_val, b_val)
    cur_sum = builder.load(sum_slot)
    new_sum = builder.fadd(cur_sum, prod)
    builder.store(new_sum, sum_slot)
    
    # k++
    k_next = builder.add(k_val, ir.Constant(i64, 1))
    builder.store(k_next, k_phi)
    builder.branch(loop_k)
    
    # After k loop - store result C[i,j] = sum
    builder.position_at_end(after_k)
    sum_final = builder.load(sum_slot)
    
    # C[i,j] = C_ptr[i*Cs0 + j*Cs1]
    i_Cs0 = builder.mul(i_val, Cs0)
    j_Cs1 = builder.mul(j_val, Cs1)
    c_index = builder.add(i_Cs0, j_Cs1)
    c_ptr = builder.gep(C_ptr, [c_index])
    builder.store(sum_final, c_ptr)
    
    # j++
    j_next = builder.add(j_val, ir.Constant(i64, 1))
    builder.store(j_next, j_phi)
    builder.branch(loop_j)
    
    # After j loop - i++
    builder.position_at_end(after_j)
    i_next = builder.add(i_val, ir.Constant(i64, 1))
    builder.store(i_next, i_phi)
    builder.branch(loop_i)
    
    # After i loop - continue to next block
    builder.position_at_end(after_i)

def generate_matmul_ir(dot_node, leaf_arrays, output_dtype, output_shape, func_name, index_map):
    """Generate IR for pure matrix multiplication"""    
    module = ir.Module(name=func_name)
    module.triple = get_default_triple()
    
    # Create function signature: void func(array_desc* A, array_desc* B, array_desc* C)
    array_desc_ptr_type = array_descriptor_type().as_pointer()
    func_type = ir.FunctionType(ir.VoidType(), [array_desc_ptr_type, array_desc_ptr_type, array_desc_ptr_type])
    func = ir.Function(module, func_type, name=func_name)
    
    A_arg, B_arg, C_arg = func.args
    
    # Create entry block
    entry = func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    
    # Generate matrix multiplication
    elem_type = dtype_to_llvm(output_dtype)
    generate_matmul_inline(builder, func, A_arg, B_arg, C_arg, elem_type)
    
    # Return
    builder.ret_void()
    
    return str(module)