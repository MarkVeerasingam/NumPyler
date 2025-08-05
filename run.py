import time 
import numpy as np
from numpyler.compile import compile

@compile
def multiply_fn(a, b):
    return np.multiply(a,b)

a = np.array([1,2,3,4], dtype=np.int32)
b = np.array([2,5,10,20], dtype=np.int32)
c = multiply_fn(a,b)
print(c)