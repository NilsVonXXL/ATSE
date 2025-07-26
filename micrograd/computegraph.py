from copy import copy
from micrograd.engine import Value


def rerun(output: Value, inputs: dict[Value, float], return_all: bool = False) -> Value:
    """Re-executes a compute graph with different input values."""
    env = copy(inputs)

    for v in output.compute_graph():
        if v in env:
            continue
    
        if len(v.prev) == 0:
            env[v] = v.data
        else:
            env[v] = eval_rules[v.op](*[env[p] for p in v.prev])
    return env[output] if not return_all else env

eval_rules = {
    "+": lambda a, b: a + b,
    "*": lambda a, b: a * b,
    "ReLU": lambda a: max(0, a),
}
