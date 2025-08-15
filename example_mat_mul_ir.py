from llvmlite import ir, binding
import numpy as np

# ---------------
# LLVM Setup
# ---------------
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

# Create a module and function type
module = ir.Module(name="matmul_module")

# double*, double*, double*, int, int, int
double_ptr = ir.DoubleType().as_pointer()
int_type = ir.IntType(32)
func_ty = ir.FunctionType(ir.VoidType(),
                          [double_ptr, double_ptr, double_ptr,
                           int_type, int_type, int_type])

matmul_fn = ir.Function(module, func_ty, name="matmul")

# Name function arguments
a_ptr, b_ptr, c_ptr, M, N, K = matmul_fn.args
a_ptr.name, b_ptr.name, c_ptr.name = "A", "B", "C"
M.name, N.name, K.name = "M", "N", "K"

# Entry block
entry = matmul_fn.append_basic_block(name="entry")
builder = ir.IRBuilder(entry)

# Outer loop over i
i_var = builder.alloca(int_type, name="i")
builder.store(ir.Constant(int_type, 0), i_var)

loop_i = matmul_fn.append_basic_block("loop_i")
after_i = matmul_fn.append_basic_block("after_i")

builder.branch(loop_i)

builder.position_at_end(loop_i)
i_val = builder.load(i_var, name="i_val")
cond_i = builder.icmp_signed("<", i_val, M)
builder.cbranch(cond_i, loop_body_i := matmul_fn.append_basic_block("loop_body_i"), after_i)

# Body of i loop
builder.position_at_end(loop_body_i)

# Inner loop over j
j_var = builder.alloca(int_type, name="j")
builder.store(ir.Constant(int_type, 0), j_var)

loop_j = matmul_fn.append_basic_block("loop_j")
after_j = matmul_fn.append_basic_block("after_j")
builder.branch(loop_j)

builder.position_at_end(loop_j)
j_val = builder.load(j_var, name="j_val")
cond_j = builder.icmp_signed("<", j_val, N)
builder.cbranch(cond_j, loop_body_j := matmul_fn.append_basic_block("loop_body_j"), after_j)

# Body of j loop
builder.position_at_end(loop_body_j)

# sum = 0.0
sum_var = builder.alloca(ir.DoubleType(), name="sum")
builder.store(ir.Constant(ir.DoubleType(), 0.0), sum_var)

# Loop over k
k_var = builder.alloca(int_type, name="k")
builder.store(ir.Constant(int_type, 0), k_var)

loop_k = matmul_fn.append_basic_block("loop_k")
after_k = matmul_fn.append_basic_block("after_k")
builder.branch(loop_k)

builder.position_at_end(loop_k)
k_val = builder.load(k_var, name="k_val")
cond_k = builder.icmp_signed("<", k_val, K)
builder.cbranch(cond_k, loop_body_k := matmul_fn.append_basic_block("loop_body_k"), after_k)

# k loop body
builder.position_at_end(loop_body_k)

# A[i, k]
a_index = builder.add(builder.mul(i_val, K), k_val)
a_elem_ptr = builder.gep(a_ptr, [a_index])
a_val = builder.load(a_elem_ptr)

# B[k, j]
b_index = builder.add(builder.mul(k_val, N), j_val)
b_elem_ptr = builder.gep(b_ptr, [b_index])
b_val = builder.load(b_elem_ptr)

# sum += a_val * b_val
prod = builder.fmul(a_val, b_val)
cur_sum = builder.load(sum_var)
new_sum = builder.fadd(cur_sum, prod)
builder.store(new_sum, sum_var)

# k++
k_next = builder.add(k_val, ir.Constant(int_type, 1))
builder.store(k_next, k_var)
builder.branch(loop_k)

# After k loop: store sum into C[i, j]
builder.position_at_end(after_k)
final_sum = builder.load(sum_var)
c_index = builder.add(builder.mul(i_val, N), j_val)
c_elem_ptr = builder.gep(c_ptr, [c_index])
builder.store(final_sum, c_elem_ptr)

# j++
j_next = builder.add(j_val, ir.Constant(int_type, 1))
builder.store(j_next, j_var)
builder.branch(loop_j)

# After j loop
builder.position_at_end(after_j)
i_next = builder.add(i_val, ir.Constant(int_type, 1))
builder.store(i_next, i_var)
builder.branch(loop_i)

# After i loop
builder.position_at_end(after_i)
builder.ret_void()

# ---------------
# JIT compilation
# ---------------
target = binding.Target.from_default_triple()
target_machine = target.create_target_machine()
backing_mod = binding.parse_assembly("")
engine = binding.create_mcjit_compiler(backing_mod, target_machine)

llvm_ir = str(module)
print(llvm_ir)
mod = binding.parse_assembly(llvm_ir)
mod.verify()
engine.add_module(mod)
engine.finalize_object()

# ---------------
# Run it
# ---------------
import ctypes
matmul_ptr = engine.get_function_address("matmul")
matmul_cfunc = ctypes.CFUNCTYPE(None,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_int, ctypes.c_int, ctypes.c_int)(matmul_ptr)

# Example data
M_val, N_val, K_val = 2, 3, 4
A_np = np.random.rand(M_val, K_val)
B_np = np.random.rand(K_val, N_val)
C_np = np.zeros((M_val, N_val), dtype=np.float64)

matmul_cfunc(A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
             B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
             C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
             M_val, N_val, K_val)

print("NumPy result:\n", A_np @ B_np)
print("LLVM result:\n", C_np)
