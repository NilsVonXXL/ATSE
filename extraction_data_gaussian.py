import itertools
import numpy as np
import pickle
from micrograd.engine import Value
from micrograd.ibp import Interval
from micrograd.branch_and_bound import branch_and_bound
import os
import json
import glob
from tqdm import tqdm

def input_folder_name(x, y, eps):
    return f"input-x-{x}-y-{y}-eps-{eps}"

def parse_model_info(model_path):
    basename = os.path.basename(model_path)
    parts = basename.replace(".pkl", "").split("_")
    dataset = parts[1]
    number = parts[2]
    return dataset, number

# Input grid
x_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
y_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
eps_vals = np.round(np.arange(0.15, 0.41, 0.1), 2)
input_combinations = list(itertools.product(x_vals, y_vals, eps_vals))

# Only handle model_gaussian
model_paths = glob.glob("models/model_gaussian_*.pkl")
for model_path in tqdm(model_paths, desc="Models"):
    dataset, number = parse_model_info(model_path)
    
    net_folder = os.path.join("gaussian", f"data-{dataset}", f"{number}")
    os.makedirs(net_folder, exist_ok=True)

    # Load model and save weights
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    weights = [p for p in model.parameters()]
    with open(os.path.join(net_folder, "parameters.pkl"), "wb") as f:
        pickle.dump(weights, f)

    for idx, (x, y, eps) in enumerate(input_combinations):
        input_vals = [x, y]
        input_vars = [Value(val) for val in input_vals]
        in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in input_vars}
        score = model(input_vars)
        if not (0 <= score.data <= 1.75):
            continue

        best_lb, best_ub, minimizer, branch_lp_bounds, relu_indexes_list = branch_and_bound(score, in_bounds)
        if not branch_lp_bounds:
            continue

        input_folder = os.path.join(net_folder, input_folder_name(x, y, eps))
        os.makedirs(input_folder, exist_ok=True)

        info = {
            'in_bounds': {str(i): (b.lower, b.upper) for i, b in in_bounds.items()},
            'best_lb': best_lb,
            'best_ub': best_ub,
            'minimizer': minimizer,
        }
        with open(os.path.join(input_folder, "info.pkl"), "wb") as f:
            pickle.dump(info, f)

        for step_idx, (step_bounds, relu_indexes) in enumerate(zip(branch_lp_bounds, relu_indexes_list)):
            step_folder = os.path.join(input_folder, f"step_{step_idx}")
            os.makedirs(step_folder, exist_ok=True)
            with open(os.path.join(step_folder, "relu_nodes.json"), "w") as f:
                json.dump(relu_indexes, f)
            branching_table = []
            for relu_idx, relu_info in enumerate(step_bounds):
                entry = {
                    "relu_index": relu_idx,
                    "split_left_lb": relu_info["split1_lb"],
                    "split_left_ub": relu_info["split1_bounds"][1],
                    "split_right_lb": relu_info["split2_lb"],
                    "split_right_ub": relu_info["split2_bounds"][1],
                }
                branching_table.append(entry)
            with open(os.path.join(step_folder, "strong_branching.pkl"), "wb") as f:
                pickle.dump(branching_table, f)
