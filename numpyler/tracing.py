import numpy as np

class TraceNode:
    def __init__(self, op_name, inputs, kwargs, result=None):
        self.op_name = op_name          # e.g., 'multiply', 'add'
        self.inputs = inputs            # list of TracedArray or scalar
        self.kwargs = kwargs            # optional keyword args
        self.result = result            # reference to output TracedArray

    def __repr__(self):
        return f"TraceNode({self.op_name}, inputs={self.inputs})"

    def dump(self, indent=0):
        pad = "  " * indent
        print(f"{pad}Op: {self.op_name}")
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, TracedArray) and inp.trace_node:
                print(f"{pad}  Input[{i}]:")
                inp.trace_node.dump(indent + 2)
            else:
                print(f"{pad}  Input[{i}]: {inp}")


class TracedArray:
    def __init__(self, data, trace_node=None):
        self.data = data
        self.trace_node = trace_node

    def __mul__(self, other):
        print(f"[TRACE] __mul__ called with {self.data} * {other}")
        return np.multiply(self, other)

    def __add__(self, other):
        print(f"[TRACE] __add__ called with {self.data} + {other}")
        return np.add(self, other)

    def __sub__(self, other):
        print(f"[TRACE] __sub__ called with {self.data} - {other}")
        return np.subtract(self, other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        def unwrap(x):
            return x.data if isinstance(x, TracedArray) else x

        # Extract actual data to compute result
        raw_inputs = [unwrap(x) for x in inputs]
        result_data = ufunc(*raw_inputs, **kwargs)

        print(f"[TRACE] ufunc {ufunc.__name__} called with inputs: {inputs}")

        # Create trace node and new traced array
        node = TraceNode(op_name=ufunc.__name__, inputs=inputs, kwargs=kwargs)
        result = TracedArray(result_data, trace_node=node)
        node.result = result
        return result

    def realize(self):
        # Convert to array if it's a scalar, otherwise just return
        if isinstance(self.data, (int, float, np.integer, np.floating)):
            return np.array([self.data], dtype=np.result_type(self.data))
        return self.data


def dump_trace(node):
    """Utility function to print a full trace tree"""
    if isinstance(node, TracedArray):
        if node.trace_node:
            print("[TRACE] Dumping trace tree:")
            node.trace_node.dump()
        else:
            print("[TRACE] No trace found.")
    elif isinstance(node, TraceNode):
        print("[TRACE] Dumping trace node:")
        node.dump()
    else:
        print("[TRACE] Not a traced object.")
