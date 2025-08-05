# numpyler/runtime.py
import numpy as np
from ctypes import c_void_p, c_longlong, Structure, POINTER

class MemRefDescriptor(Structure):
    _fields_ = [
        ("allocated", c_void_p),
        ("aligned", c_void_p),
        ("offset", c_longlong),
        ("shape", c_longlong * 1),
        ("strides", c_longlong * 1),
    ]

def numpy_to_memref(arr: np.ndarray) -> MemRefDescriptor:
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    
    desc = MemRefDescriptor()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated  # For contiguous arrays, same as allocated
    desc.offset = 0
    desc.shape[0] = arr.shape[0]
    desc.strides[0] = arr.strides[0] // arr.itemsize  # Convert to element stride
    
    # Keep a reference to the numpy array to prevent garbage collection
    desc._array_ref = arr
    return desc