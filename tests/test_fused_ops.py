import numpy as np
from numpyler import compile

# Prepare test data
test_data = {
    'int32': {
        'a': np.array([1, 2, 3], dtype=np.int32),
        'b': np.array([4, 5, 6], dtype=np.int32),
        'c': np.array([2, 3, 4], dtype=np.int32),
        'd': np.array([1, 1, 1], dtype=np.int32)
    },
    'float32': {
        'a': np.array([1, 2, 3], dtype=np.float32),
        'b': np.array([4, 5, 6], dtype=np.float32),
        'c': np.array([2, 3, 4], dtype=np.float32),
        'd': np.array([1, 1, 1], dtype=np.float32)
    }
}

# --- Basic operations ---
print("=== Basic operations ===")
data = test_data['int32']

@compile
def ops1(a, b, c):
    return a + b * c

@compile
def ops2(a, b, c):
    return (a + b) * c

result1 = ops1(data['a'], data['b'], data['c'])
print("ops1:", result1)

result2 = ops2(data['a'], data['b'], data['c'])
print("ops2:", result2)

# --- Mixed operations ---
print("\n=== Mixed operations ===")
data = test_data['float32']

@compile
def ops(a, b, c, d):
    return a * b + c / d

result = ops(data['a'], data['b'], data['c'], data['d'])
print("mixed ops:", result)

# --- Float operations ---
print("\n=== Float operations ===")
data = test_data['float32']

@compile
def ops(a, b):
    return a * 1.5 + b / 2.0

result = ops(data['a'], data['b'])
print("float ops:", result)

# --- Scalar operations ---
print("\n=== Scalar operations ===")
data = test_data['int32']

@compile
def ops(a):
    return a * 2 + 5

result = ops(data['a'])
print("scalar ops:", result)

# --- Chained operations ---
print("\n=== Chained operations ===")
data = test_data['int32']

@compile
def ops(a, b, c):
    return a - b + c * 2

result = ops(data['a'], data['b'], data['c'])
print("chained ops:", result)

# --- Cache behavior ---
print("\n=== Cache behavior ===")
data = test_data['int32']

@compile
def ops(a, b):
    return a + b

result1 = ops(data['a'], data['b'])
result2 = ops(data['a'], data['b'])
print("cache first:", result1)
print("cache second:", result2)

# --- Shape mismatch ---
print("\n=== Shape mismatch ===")
data = test_data['int32']
b_wrong = np.array([4, 5], dtype=np.int32)

@compile
def ops(a, b):
    return a + b

try:
    ops(data['a'], b_wrong)
except ValueError as e:
    print("shape mismatch error:", e)

# --- Empty array ---
print("\n=== Empty array ===")
empty = np.array([], dtype=np.int32)

@compile
def ops(a, b):
    return a + b

result = ops(empty, empty)
print("empty array result:", result, "shape:", result.shape, "dtype:", result.dtype)
