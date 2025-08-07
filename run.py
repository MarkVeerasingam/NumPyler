# run.py
import numpy as np
from numpyler.compile import compile

# Test 1: Simple fused operation
@compile
def fused_ops1(a, b, c):
    return a + b * c  # [1+4*2, 2+5*3, 3+6*4] = [9, 17, 27]

# Test 2: Multiple operations with different precedence
@compile
def fused_ops2(a, b, c):
    return (a + b) * c # [(1+4)*2, (2+5)*3, (3+6)*4] = [10, 21, 36]

# Test 3: Complex expression with multiple operations
@compile
def fused_ops3(a, b, c, d):
    return a * b + c / d # [1*4+2/1, 2*5+3/1, 3*6+4/1] = [4+2, 10+3, 18+4] = [6, 13, 22]

# Test 4: Expression with constants
@compile
def fused_ops4(a):
    return a * 2 + 5  # [1*2+5, 2*2+5, 3*2+5] = [7, 9, 11]

# Test 5: Multiple chained operations
@compile
def fused_ops5(a, b, c):
    return a - b + c * 2  # [1-4+2*2, 2-5+3*2, 3-6+4*2] = [-3+4, -3+6, -3+8] = [1, 3, 5]

# Test 6: Mixed types (float operations)
@compile
def fused_ops6(a, b):
    return a * 1.5 + b / 2.0  # [1.5+4/2, 3.0+5/2, 4.5+6/2] = [1.5+2, 3.0+2.5, 4.5+3] = [3.5, 5.5, 7.5]

# Input arrays
a = np.array([1, 2, 3], dtype=np.int64)
b = np.array([4, 5, 6], dtype=np.int64)
c = np.array([2, 3, 4], dtype=np.int64)
d = np.array([1, 1, 1], dtype=np.int64)

# Run tests
print("Test 1: a + b * c =", fused_ops1(a, b, c))
print("Test 2: (a + b) * c =", fused_ops2(a, b, c))
print("Test 3: a * b + c / d =", fused_ops3(a, b, c, d))
print("Test 4: a * 2 + 5 =", fused_ops4(a))
print("Test 5: a - b + c * 2 =", fused_ops5(a, b, c))
print("Test 6: a * 1.5 + b / 2.0 =", fused_ops6(a.astype(np.float32), b.astype(np.float32)))
