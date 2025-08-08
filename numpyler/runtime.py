import numpy as np
from ctypes import c_void_p, c_longlong, Structure

MAX_DIMS = 8  # Support up to 8 dimensions

class MemRefDescriptor(Structure):
    _fields_ = [
        ("allocated", c_void_p),
        ("aligned", c_void_p),
        ("offset", c_longlong),
        ("rank", c_longlong),  # Number of dimensions
        ("shape", c_longlong * MAX_DIMS),
        ("stride", c_longlong * MAX_DIMS),
    ]

def numpy_to_memref(arr: np.ndarray) -> MemRefDescriptor:
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    
    if arr.ndim > MAX_DIMS:
        raise ValueError(f"Array has {arr.ndim} dimensions, but only up to {MAX_DIMS} are supported")
    
    desc = MemRefDescriptor()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    desc.rank = arr.ndim
    
    # Set shape and strides
    for i in range(arr.ndim):
        desc.shape[i] = arr.shape[i]
        desc.stride[i] = arr.strides[i] // arr.itemsize  # Convert byte strides to element strides
    
    # Zero out unused dimensions
    for i in range(arr.ndim, MAX_DIMS):
        desc.shape[i] = 0
        desc.stride[i] = 0
    
    return desc