import numpy as np
import pickle
from micrograd.engine import Value
from micrograd.ibp import Interval
from micrograd.branch_and_bound import branch_and_bound
import glob
import itertools
import os
import json


x_vals = np.round(np.arange(-1.0, 1.01, 0.1), 2)
y_vals = np.round(np.arange(-1.0, 1.01, 0.1), 2)
eps_vals = np.round(np.arange(0.1, 0.51, 0.1), 2)

input_combinations = list(itertools.product(x_vals, y_vals, eps_vals))

def input_folder_name(x, y, eps):
    return f"x-{x}-y-{y}-eps-{eps}"



# List your datasets and model paths
datasets = ["moons", "circles", "blobs", "classification", "gaussian"]
model_paths = glob.glob("models/*.pkl")

# Map each model to its dataset and number (assuming naming: model_{dataset}_{number}.pkl)
def parse_model_info(model_path):
    # Example: models/model_moons_1.pkl
    basename = os.path.basename(model_path)
    parts = basename.replace(".pkl", "").split("_")
    dataset = parts[1]
    number = parts[2]
    return dataset, number

for model_path in model_paths:
    dataset, number = parse_model_info(model_path)
    # Create dataset and network folders
    net_folder = f"data-{dataset}/{number}"
    os.makedirs(net_folder, exist_ok=True)
    # Load model and save weights
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    weights = [p.data for p in model.parameters()]
    with open(os.path.join(net_folder, "parameters.pkl"), "wb") as f:
        pickle.dump(weights, f)

    # For each input/epsilon combination
    for x, y, eps in input_combinations:
        input_folder = os.path.join(net_folder, input_folder_name(x, y, eps))
        os.makedirs(input_folder, exist_ok=True)
        # Prepare input and bounds
        input_vals = [x, y]
        input_vars = [Value(val) for val in input_vals]
        in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in input_vars}
        score = model(input_vars)
        # Run branch-and-bound
        best_lb, best_ub, minimizer, branch_lp_bounds = branch_and_bound(score, in_bounds)
        # Save info.pkl
        info = {
            'in_bounds': {str(i): (b.lower, b.upper) for i, b in in_bounds.items()},
            'best_lb': best_lb,
            'best_ub': best_ub,
            'minimizer': minimizer,
        }
        with open(os.path.join(input_folder, "info.pkl"), "wb") as f:
            pickle.dump(info, f)

        # Save branching step data
        for step_idx, step_bounds in enumerate(branch_lp_bounds):
            step_folder = os.path.join(input_folder, f"step_{step_idx}")
            os.makedirs(step_folder, exist_ok=True)

            # relu_nodes.json: relu indices and split status
            relu_status = {str(i): -1 for i in range(len(step_bounds))}  # Default: not split
            # If you have split info, update +1 for split nodes as needed

            with open(os.path.join(step_folder, "relu_nodes.json"), "w") as f:
                json.dump(relu_status, f)

            # strong_branching.pkl: table for each relu node
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
