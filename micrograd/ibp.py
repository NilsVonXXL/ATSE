from dataclasses import dataclass
from copy import copy
from micrograd.engine import Value


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __iter__(self):
        yield self.lower
        yield self.upper
    


def ibp(output: Value, in_bounds: dict[Value, Interval], return_all: bool = False) -> Interval:
    env = copy(in_bounds)

    for v in output.compute_graph():
        input = False
        if v in env:
            input = True
            continue
            
        if len(v.prev) == 0:
            env[v] = Interval(v.data, v.data)
        else:
            env[v] = ibp_rules[v.op](*[env[p] for p in v.prev])
    return env[output] if not return_all else env


ibp_rules = {
    "+": lambda a, b: Interval(a.lower + b.lower, a.upper + b.upper),
    "ReLU": lambda a: Interval(max(0, a.lower), max(0, a.upper)),
}


def _ibp_mul_rule(a: Interval, b: Interval) -> Interval:
    products = [
        a.lower * b.lower,
        a.lower * b.upper,
        a.upper * b.lower,
        a.upper * b.upper,
    ]
    return Interval(min(products), max(products))

ibp_rules["*"] = _ibp_mul_rule

