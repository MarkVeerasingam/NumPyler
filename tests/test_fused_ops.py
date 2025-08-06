# test_fused_ops.py
import pytest
import numpy as np
from numpyler.compile import compile

@pytest.fixture
def test_data():
    return {
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

def test_basic_operations(test_data):
    data = test_data['int32']
    
    @compile
    def ops1(a, b, c):
        return a + b * c
    
    @compile
    def ops2(a, b, c):
        return (a + b) * c
    
    result1 = ops1(data['a'], data['b'], data['c'])
    expected1 = np.array([1+4*2, 2+5*3, 3+6*4], dtype=np.int32)
    np.testing.assert_array_equal(result1, expected1)
    
    result2 = ops2(data['a'], data['b'], data['c'])
    expected2 = np.array([(1+4)*2, (2+5)*3, (3+6)*4], dtype=np.int32)
    np.testing.assert_array_equal(result2, expected2)

def test_mixed_operations(test_data):
    data = test_data['float32']  # Use float32 for division
    
    @compile
    def ops(a, b, c, d):
        return a * b + c / d
    
    result = ops(data['a'], data['b'], data['c'], data['d'])
    expected = np.array([1*4+2/1, 2*5+3/1, 3*6+4/1], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_float_operations(test_data):
    data = test_data['float32']
    
    @compile
    def ops(a, b):
        return a * 1.5 + b / 2.0
    
    result = ops(data['a'], data['b'])
    expected = np.array([1.5+4/2, 3.0+5/2, 4.5+6/2], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_scalar_operations(test_data):
    data = test_data['int32']
    
    @compile
    def ops(a):
        return a * 2 + 5
    
    result = ops(data['a'])
    expected = np.array([1*2+5, 2*2+5, 3*2+5], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

def test_chained_operations(test_data):
    data = test_data['int32']
    
    @compile
    def ops(a, b, c):
        return a - b + c * 2
    
    result = ops(data['a'], data['b'], data['c'])
    expected = np.array([1-4+2*2, 2-5+3*2, 3-6+4*2], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

def test_type_promotion():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.float32)
    
    @compile
    def ops(a, b):
        return a * b
    
    result = ops(a, b)
    assert result.dtype == np.float32  # Should promote to float32
    expected = np.array([1*4, 2*5, 3*6], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

def test_cache_behavior(test_data):
    data = test_data['int32']
    
    @compile
    def ops(a, b):
        return a + b
    
    result1 = ops(data['a'], data['b'])
    result2 = ops(data['a'], data['b'])
    expected = data['a'] + data['b']
    np.testing.assert_array_equal(result1, expected)
    np.testing.assert_array_equal(result2, expected)

def test_unsupported_operation(test_data):
    data = test_data['int32']
    
    @compile
    def ops(a, b):
        return a ** b  # Power operation not supported
    
    with pytest.raises(ValueError, match="Power operation not supported"):
        ops(data['a'], data['b'])

def test_shape_mismatch(test_data):
    data = test_data['int32']
    b_wrong = np.array([4, 5], dtype=np.int32)
    
    @compile
    def ops(a, b):
        return a + b
    
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        ops(data['a'], b_wrong)

def test_empty_array():
    empty = np.array([], dtype=np.int32)
    
    @compile
    def ops(a, b):
        return a + b
    
    result = ops(empty, empty)
    assert result.shape == (0,)
    assert result.dtype == np.int32