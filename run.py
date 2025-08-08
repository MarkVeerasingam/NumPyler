import numpy as np
from numpyler import compile

@compile
def eq(a, b, c, d):
    return np.add(a, np.multiply(b,c)) / d

a = np.array([1,2,4,8], dtype=np.float64)
b = np.array([2,4,8,16], dtype=np.float64)
c = np.array([3,6,12,24], dtype=np.float64)
d = np.array([4,8,16,32], dtype=np.float64)

print(eq(a,b,c,d))