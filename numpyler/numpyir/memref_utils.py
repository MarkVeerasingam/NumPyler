# numpyler/numpyir/memref_utils.py
from llvmlite import ir

def memref_type():
    i8ptr = ir.IntType(8).as_pointer()
    i64 = ir.IntType(64)
    sizes = ir.ArrayType(i64, 8)
    strides = ir.ArrayType(i64, 8)
    return ir.LiteralStructType([i8ptr, i8ptr, i64, i64, sizes, strides])

def get_aligned_ptr(builder, memref_arg):
    i32 = ir.IntType(32)
    aligned_ptr_ptr = builder.gep(memref_arg, [ir.Constant(i32, 0), ir.Constant(i32, 1)])
    return builder.load(aligned_ptr_ptr)

def get_size(builder, memref_arg, dim):
    i32 = ir.IntType(32)
    sizes_gep = builder.gep(memref_arg, [ir.Constant(i32, 0), ir.Constant(i32, 4)])
    elem_gep = builder.gep(sizes_gep, [ir.Constant(i32, 0), ir.Constant(i32, dim)])
    return builder.load(elem_gep)

def get_stride(builder, memref_arg, dim):
    i32 = ir.IntType(32)
    strides_gep = builder.gep(memref_arg, [ir.Constant(i32, 0), ir.Constant(i32, 5)])
    elem_gep = builder.gep(strides_gep, [ir.Constant(i32, 0), ir.Constant(i32, dim)])
    return builder.load(elem_gep)

def dtype_to_llvm(dtype):
    import numpy as np
    if dtype == np.int32:
        return ir.IntType(32)
    elif dtype == np.int64:
        return ir.IntType(64)
    elif dtype == np.float32:
        return ir.FloatType()
    elif dtype == np.float64:
        return ir.DoubleType()
    raise ValueError(f"Unsupported dtype: {dtype}")