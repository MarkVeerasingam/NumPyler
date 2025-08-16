# numpyler/runtime.py
import numpy as np
from ctypes import c_void_p, c_longlong, Structure, POINTER, c_int64, c_double, c_float, c_int32

class ArrayDescriptor(Structure):
    """Simplified array descriptor that directly exposes NumPy array properties"""
    _fields_ = [
        ("data", c_void_p),
        ("size", c_longlong),
        ("ndim", c_longlong),
        ("shape", POINTER(c_longlong)),
        ("strides", POINTER(c_longlong)),
    ]

def numpy_to_descriptor(arr: np.ndarray) -> ArrayDescriptor:
    """Convert NumPy array to simplified descriptor"""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    
    desc = ArrayDescriptor()
    desc.data = arr.ctypes.data_as(c_void_p)
    desc.size = arr.size
    desc.ndim = arr.ndim
    
    # Convert shape and strides to ctypes arrays
    shape_array = (c_longlong * arr.ndim)(*arr.shape)
    stride_array = (c_longlong * arr.ndim)(*(s // arr.itemsize for s in arr.strides))
    
    desc.shape = shape_array
    desc.strides = stride_array
    
    return desc, shape_array, stride_array