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


def test_branch_and_bound(moons_model):
    x = [Value(0), Value(0.5)]
    in_bounds = {xi: Interval(xi.data - 0.1, xi.data + 0.1) for xi in x}
    out = moons_model(x)
    lower_bound, upper_bound = branch_and_bound(out, in_bounds)
    print(lower_bound, upper_bound)


def test_ibp(moons_model):
    x = [Value(1), Value(-1)]
    in_bounds = {xi: Interval(xi.data - 0.1, xi.data + 0.1) for xi in x}
    out = moons_model(x)
    score = ibp(out, in_bounds)
    print(score)


if __name__ == "__main__":
    pytest.run()
