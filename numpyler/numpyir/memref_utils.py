# numpyler/numpyir/array_utils.py
from llvmlite import ir

def array_descriptor_type():
    """Define the LLVM struct type for our simplified array descriptor"""
    i8ptr = ir.IntType(8).as_pointer()
    i64 = ir.IntType(64)
    i64ptr = i64.as_pointer()
    
    return ir.LiteralStructType([
        i8ptr,    # data
        i64,      # size  
        i64,      # ndim
        i64ptr,   # shape
        i64ptr,   # strides
    ])

def get_data_ptr(builder, array_arg):
    """Get the data pointer from array descriptor"""
    i32 = ir.IntType(32)
    data_ptr_ptr = builder.gep(array_arg, [ir.Constant(i32, 0), ir.Constant(i32, 0)])
    return builder.load(data_ptr_ptr)

def get_size(builder, array_arg):
    """Get the total size from array descriptor"""
    i32 = ir.IntType(32)
    size_ptr = builder.gep(array_arg, [ir.Constant(i32, 0), ir.Constant(i32, 1)])
    return builder.load(size_ptr)

def get_ndim(builder, array_arg):
    """Get the number of dimensions from array descriptor"""
    i32 = ir.IntType(32)
    ndim_ptr = builder.gep(array_arg, [ir.Constant(i32, 0), ir.Constant(i32, 2)])
    return builder.load(ndim_ptr)

def get_shape_ptr(builder, array_arg):
    """Get pointer to shape array from array descriptor"""
    i32 = ir.IntType(32)
    shape_ptr_ptr = builder.gep(array_arg, [ir.Constant(i32, 0), ir.Constant(i32, 3)])
    return builder.load(shape_ptr_ptr)

def get_strides_ptr(builder, array_arg):
    """Get pointer to strides array from array descriptor"""
    i32 = ir.IntType(32)
    strides_ptr_ptr = builder.gep(array_arg, [ir.Constant(i32, 0), ir.Constant(i32, 4)])
    return builder.load(strides_ptr_ptr)

def get_shape_dim(builder, array_arg, dim):
    """Get shape for specific dimension"""
    i32 = ir.IntType(32)
    i64 = ir.IntType(64)
    shape_ptr = get_shape_ptr(builder, array_arg)
    dim_ptr = builder.gep(shape_ptr, [ir.Constant(i64, dim)])
    return builder.load(dim_ptr)

def get_stride_dim(builder, array_arg, dim):
    """Get stride for specific dimension"""
    i32 = ir.IntType(32)
    i64 = ir.IntType(64)
    strides_ptr = get_strides_ptr(builder, array_arg)
    dim_ptr = builder.gep(strides_ptr, [ir.Constant(i64, dim)])
    return builder.load(dim_ptr)

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