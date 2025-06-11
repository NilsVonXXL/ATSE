import pickle
from micrograd.engine import Value
from micrograd.ibp import Interval, ibp
from micrograd.branch_and_bound import branch_and_bound  # Import your main function
import os

def test_branch_and_bound():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    x = [Value(0), Value(0.5)]
    in_bounds = {xi: Interval(xi.data - 0.1, xi.data + 0.1) for xi in x}
    out = loaded_model(x)
    lower_bound, upper_bound = branch_and_bound(out, in_bounds)
    return lower_bound, upper_bound

def test_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    x = [Value(1), Value(-1)]
    in_bounds = {xi: Interval(xi.data - 0.1, xi.data + 0.1) for xi in x}
    out = loaded_model(x)
    score = ibp(out, in_bounds)
    return score


print(test_branch_and_bound())
#print(test_model())
