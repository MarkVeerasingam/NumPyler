import numpy as np
from ctypes import c_void_p, c_longlong, Structure

class MemRefDescriptor(Structure):
    _fields_ = [
        ("allocated", c_void_p),
        ("aligned", c_void_p),
        ("offset", c_longlong),
        ("shape", c_longlong * 1),
        ("stride", c_longlong * 1),
    ]

def numpy_to_memref(arr: np.ndarray) -> MemRefDescriptor:
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    desc = MemRefDescriptor()
    desc.allocated = arr.ctypes.data_as(c_void_p)
    desc.aligned = desc.allocated
    desc.offset = 0
    desc.shape[0] = arr.shape[0]
    desc.stride[0] = 1
    return desc
