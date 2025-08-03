import pickle
from micrograd.engine import Value
from micrograd.ibp import Interval, ibp
from micrograd.branch_and_bound import branch_and_bound  # Import your main function
from pathlib import Path

import pytest


@pytest.fixture
def moons_model():
    resource_dir = Path(__file__).parent.parent / "resources"
    model_path = resource_dir / "moons.pkl"
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


@pytest.mark.parametrize("x1,x2", [(0.0, 0.5), (-0.5, 1.0), (2.0, -0.5)])
@pytest.mark.parametrize("eps", [0.1, 0.5])
def test_branch_and_bound(moons_model, x1, x2, eps):
    print("=" * 80)
    print(f"Branch and bound for x1={x1}, x2={x2}, eps={eps}")
    print("=" * 80)

    x = [Value(x1), Value(x2)]
    in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in x}
    out = moons_model(x)
    lower_bound, upper_bound, cx = branch_and_bound(out, in_bounds)

    print("Result:", lower_bound, upper_bound, cx)
    assert lower_bound <= upper_bound
    assert lower_bound <= out.data
    if cx is None:
        assert lower_bound >= 0
    else:
        cx = [cx[xi] for xi in x]
        out = moons_model(cx)
        assert out.data < 0


def test_ibp(moons_model):
    x = [Value(1), Value(-1)]
    in_bounds = {xi: Interval(xi.data - 0.1, xi.data + 0.1) for xi in x}
    out = moons_model(x)
    score = ibp(out, in_bounds)
    print(score)


if __name__ == "__main__":
    pytest.run()
