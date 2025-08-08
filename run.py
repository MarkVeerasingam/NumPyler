import numpy as np
from numpyler import compile

@compile
def tensor_sum(a, b, c, d):
    return np.add(a, np.multiply(b,c)) / d

a = np.array([[1.5, 2.1], [3.8, 4.6]], dtype=np.float64)
b = np.array([[2.9, 3.6], [4.4, 5.6]], dtype=np.float64)
c = np.array([[5.2, 6.5], [7.1, 8.4]], dtype=np.float64)
d = np.array([[1.8, 2.4], [3.6, 4.7]], dtype=np.float64)

print(tensor_sum(a,b,c,d))