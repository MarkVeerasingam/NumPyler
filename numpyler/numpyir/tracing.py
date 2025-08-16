# numpyler/numpyir/tracing.py
import numpy as np

class TraceNode:
    def __init__(self, op_name, inputs, kwargs, result=None):
        self.op_name = op_name
        self.inputs = inputs
        self.kwargs = kwargs
        self.result = result
        self.id = id(self)

    def __repr__(self):
        return f"TraceNode({self.op_name}, inputs={self.inputs})"

    def dump(self, indent=0):
        pad = "  " * indent
        print(f"{pad}Op: {self.op_name}")
        for i, inp in enumerate(self.inputs):
            if isinstance(inp, TracedArray) and inp.trace_node is not None:
                print(f"{pad}  Input[{i}]")
                inp.trace_node.dump(indent + 2)
            else:
                print(f"{pad}  Input[{i}]: {inp}")

class TracedArray:
    def __init__(self, data, trace_node=None, original_index=None):
        self.data = data
        self.trace_node = trace_node
        self.original_index = original_index
        self.id = id(self)
        
    def __add__(self, other): return np.add(self, other)
    def __sub__(self, other): return np.subtract(self, other)
    def __mul__(self, other): return np.multiply(self, other)
    def __truediv__(self, other): return np.divide(self, other)
    def __rtruediv__(self, other): return np.divide(other, self)
    def __radd__(self, other): return np.add(other, self)
    def __rsub__(self, other): return np.subtract(other, self)
    def __rmul__(self, other): return np.multiply(other, self)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        
        print(f"DEBUG: ufunc={ufunc}, name={ufunc.__name__}")  
            
        # Convert non-TracedArray inputs to TracedArray
        inputs = [
            x if isinstance(x, TracedArray) else TracedArray(x)
            for x in inputs
        ]

        unwrapped_inputs, raw_inputs = [], []
        for x in inputs:
            if isinstance(x, TracedArray):
                unwrapped_inputs.append(x)
                raw_inputs.append(x.data)
            else:
                unwrapped_inputs.append(x)
                raw_inputs.append(x)

        result_data = ufunc(*raw_inputs, **kwargs)
        node = TraceNode(ufunc.__name__, unwrapped_inputs, kwargs)
        result_traced = TracedArray(result_data, trace_node=node)
        node.result = result_traced
        return result_traced
    
    def __array_function__(self, func, types, args, kwargs):
        if func is np.dot:
            print("DEBUG: Found np.dot operation")
            a, b = args[0], args[1]
            # Compute result data for shape/dtype info
            a_data = a.data if isinstance(a, TracedArray) else a
            b_data = b.data if isinstance(b, TracedArray) else b
            result_data = np.dot(a_data, b_data)
            
            # Create trace node
            node = TraceNode('dot', [a, b], kwargs)
            return TracedArray(result_data, trace_node=node)
        return NotImplemented

    def realize(self):
        if isinstance(self.data, (int, float, np.integer, np.floating)):
            return np.array([self.data], dtype=np.result_type(self.data))
        return self.data

def collect_nodes(root_node):
    nodes, visited = [], set()
    def visit(node):
        if node.id in visited:
            return
        visited.add(node.id)
        for inp in node.inputs:
            if isinstance(inp, TracedArray) and inp.trace_node is not None:
                visit(inp.trace_node)
        nodes.append(node)
    if isinstance(root_node, TracedArray) and root_node.trace_node:
        visit(root_node.trace_node)
    return nodes

def dump_trace(node):
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