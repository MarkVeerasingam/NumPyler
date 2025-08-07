import numpy as np
import timeit
from numpyler.compile import compile

# Compiled functions

@compile
def fused_ops1(a, b, c):
    return a + b * c

@compile
def fused_ops2(a, b, c):
    return (a + b) * c

@compile
def fused_ops3(a, b, c, d):
    return a * b + c / d

@compile
def fused_ops4(a):
    return a * 2 + 5

@compile
def fused_ops5(a, b, c):
    return a - b + c * 2

@compile
def fused_ops6(a, b):
    return a * 1.5 + b / 2.0

# Equivalent NumPy functions

def np_ops1(a, b, c):
    return a + b * c

def np_ops2(a, b, c):
    return (a + b) * c

def np_ops3(a, b, c, d):
    return a * b + c / d

def np_ops4(a):
    return a * 2 + 5

def np_ops5(a, b, c):
    return a - b + c * 2

def np_ops6(a, b):
    return a * 1.5 + b / 2.0

# Inputs
N = 10**7
a = np.random.randint(1, 10, size=N).astype(np.float64)
b = np.random.randint(1, 10, size=N).astype(np.float64)
c = np.random.randint(1, 10, size=N).astype(np.float64)
d = np.random.randint(1, 10, size=N).astype(np.float64)

# List of tests: (name, compiled_fn, numpy_fn, input_args)
tests = [
    ("fused_ops1", fused_ops1, np_ops1, (a, b, c)),
    ("fused_ops2", fused_ops2, np_ops2, (a, b, c)),
    ("fused_ops3", fused_ops3, np_ops3, (a, b, c, d)),
    ("fused_ops4", fused_ops4, np_ops4, (a,)),
    ("fused_ops5", fused_ops5, np_ops5, (a, b, c)),
    ("fused_ops6", fused_ops6, np_ops6, (a, b)),
]

def run_benchmark(name, compiled_fn, numpy_fn, args):
    # Warm up
    compiled_fn(*args)

    compiled_time = timeit.timeit(lambda: compiled_fn(*args), number=3) / 3
    numpy_time = timeit.timeit(lambda: numpy_fn(*args), number=3) / 3

    speedup = numpy_time / compiled_time if compiled_time > 0 else float('inf')

    print(f"{name}: Compiled {compiled_time:.6f}s, NumPy {numpy_time:.6f}s, Speedup: {speedup:.2f}x")

# Run all tests immediately on script execution
for test in tests:
    run_benchmark(*test)
