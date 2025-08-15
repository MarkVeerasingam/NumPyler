import numpy as np
from numpyler import compile

@compile
def equation(a, b):
    res = np.add(a,b)
    return res

a = np.array([[1, 2], [3, 4]], dtype=np.float64)
b = np.array([[2, 3], [4, 5]], dtype=np.float64)

result = equation(a,b)
print(result)
