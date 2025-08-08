import numpy as np
import timeit
from numpyler import compile

# Compiled functions using explicit numpy ufuncs
@compile
def fused_ops1(a, b, c):
    return np.add(a, np.multiply(b, c))

@compile
def fused_ops2(a, b, c):
    return np.multiply(np.add(a, b), c)

@compile
def fused_ops3(a, b, c, d):
    return np.add(np.multiply(a, b), np.divide(c, d))

@compile
def fused_ops4(a):
    return np.add(np.multiply(a, 2), 5)

@compile
def fused_ops5(a, b, c):
    return np.add(np.subtract(a, b), np.multiply(c, 2))

@compile
def fused_ops6(a, b):
    return np.add(np.multiply(a, 1.5), np.divide(b, 2.0))

# Equivalent NumPy functions using same ufuncs
def np_ops1(a, b, c):
    return np.add(a, np.multiply(b, c))

def np_ops2(a, b, c):
    return np.multiply(np.add(a, b), c)

def np_ops3(a, b, c, d):
    return np.add(np.multiply(a, b), np.divide(c, d))

def np_ops4(a):
    return np.add(np.multiply(a, 2), 5)

def np_ops5(a, b, c):
    return np.add(np.subtract(a, b), np.multiply(c, 2))

def np_ops6(a, b):
    return np.add(np.multiply(a, 1.5), np.divide(b, 2.0))

# Inputs
N = 10**7
a = np.random.randint(1, 100, size=N).astype(np.float64)
b = np.random.randint(1, 100, size=N).astype(np.float64)
c = np.random.randint(1, 100, size=N).astype(np.float64)
d = np.random.randint(1, 100, size=N).astype(np.float64)

# Test cases
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
    numpy_fn(*args)  # Also warming up NumPy
    
    trials = 5
    compiled_times = []
    numpy_times = []
    
    for _ in range(trials):
        compiled_time = timeit.timeit(lambda: compiled_fn(*args), number=10)
        numpy_time = timeit.timeit(lambda: numpy_fn(*args), number=10)
        compiled_times.append(compiled_time)
        numpy_times.append(numpy_time)
    
    # Use median to calculate the speed up
    compiled_median = np.median(compiled_times)
    numpy_median = np.median(numpy_times)
    speedup = numpy_median / compiled_median
    
    print(f"{name}: Compiled {compiled_median:.6f}s, NumPy {numpy_median:.6f}s, Speedup: {speedup:.2f}x")

print(f"Performance Benchmark (N={N})")
print("=" * 50)
for test in tests:
    run_benchmark(*test)