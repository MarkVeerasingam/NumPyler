import numpy as np
from numpyler import compile

@compile
def equation(a, b):
    res = np.dot(a, b)
    return res

a = np.array([[1,0], [0,1]], dtype=np.float64)
b = np.array([[4,1], [2,2]], dtype=np.float64)

result = equation(a, b)
print(result)
