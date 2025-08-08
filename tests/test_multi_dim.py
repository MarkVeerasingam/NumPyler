import numpy as np
from numpyler import compile

print("\n=== Multi-dimensional operations ===")

shapes = [
    (2, 3),         # 2D small
    (3, 3),         # square matrix
    (2, 2, 3),      # 3D tensor
]

for shape in shapes:
    print(f"\n-- Testing shape {shape} --")
    a = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    b = a + 3  # just to make it different
    c = a + 1

    @compile
    def ops(a, b, c):
        return (np.multiply(a, b) + c) / 2


    result = ops(a, b, c)
    expected = (a * b + c) / 2.0  # NumPy reference

    print("input a:\n", a)
    print("input b:\n", b)
    print("input c:\n", c)
    print("result:\n", result)
    print("expected:\n", expected)
    print("match:", np.allclose(result, expected))
