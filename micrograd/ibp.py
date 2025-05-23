from dataclasses import dataclass
from copy import deepcopy
from micrograd.engine import Value


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


def ibp(output: Value, in_bounds: dict[Value, Interval]) -> Interval:
    env = deepcopy(in_bounds)

    for v in output.compute_graph():
        if len(v.prev) == 0:
            if v not in env:
                env[v] = Interval(v.data, v.data)
        else:
            env[v] = ibp_rules[v.op](*[env[p] for p in v.prev])
    return env[output]


ibp_rules = {
    "+": lambda a, b: Interval(a.lower + b.lower, a.upper + b.upper),
    "ReLU": lambda a: Interval(max(0, a.lower), max(0, a.upper)),
}


def _ibp_mul_rule(a: Interval | Value, b: Interval | Value) -> Interval:
    if isinstance(a, Value) and isinstance(b, Value):
        lb = ub = a.data * b.data

    if isinstance(b, Interval):
        a, b = b, a

    if isinstance(a, Interval) and isinstance(b, Value):
        b = b.data
        if b >= 0:
            lb = a.lower * b
            ub = a.upper * b
        else:
            lb = a.upper * b
            ub = a.lower * b
    else:
        options = [
            a.lower * b.lower,
            a.lower * b.upper,
            a.upper * b.lower,
            a.upper * b.upper,
        ]
        lb = min(options)
        ub = max(options)
    return Interval(lb, ub)


ibp_rules["*"] = _ibp_mul_rule
