import itertools
import numpy as np
import pickle
from micrograd.engine import Value
from micrograd.ibp import Interval
from micrograd.branch_and_bound import branch_and_bound
import os
import json
import sys
from tqdm import tqdm

#def find_critical_points(model, input_combinations, threshold=1.5):
#    critical_points = []
#    for x, y, eps  in input_combinations:
#        out = model([Value(x), Value(y)]).data
#        if (0 <= out <= threshold):
#            critical_points.append((x, y, eps))
#    return critical_points

def input_folder_name(x, y, eps):
    return f"test-x-{x}-y-{y}-eps-{eps}"

model_path = "models/model_moons_1.pkl"
if not os.path.exists(model_path):
    raise RuntimeError("models/model_moons_1.pkl not found!")

def parse_model_info(model_path):
    basename = os.path.basename(model_path)
    parts = basename.replace(".pkl", "").split("_")
    dataset = parts[1]
    number = parts[2]
    return dataset, number

dataset, number = parse_model_info(model_path)
net_folder = f"test-data-{dataset}/{number}"
os.makedirs(net_folder, exist_ok=True)

# Load model ONCE and save weights
with open(model_path, "rb") as f:
    model = pickle.load(f)
weights = [p for p in model.parameters()] # For GNN edge extraction
with open(os.path.join(net_folder, "test-parameters.pkl"), "wb") as f:
    pickle.dump(weights, f)

# Find critical points after loading the model
x_vals = np.round(np.arange(0, 1.51, 0.1), 2)
y_vals = np.round(np.arange(0, 1.51, 0.1), 2)
eps_vals = np.round(np.arange(0.1, 0.41, 0.1), 2)
input_combinations = list(itertools.product(x_vals, y_vals, eps_vals))


for idx, (x, y, eps) in enumerate(tqdm(input_combinations, desc="Input combinations")):
    
    input_vals = [x, y]
    input_vars = [Value(val) for val in input_vals]
    in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in input_vars}
    score = model(input_vars)
    if (0 <= score.data <= 2):
        continue

    best_lb, best_ub, minimizer, branch_lp_bounds, relu_indexes_list = branch_and_bound(score, in_bounds)

    # Only keep points with at least one branching step
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
    with open(os.path.join(input_folder, "test-info.pkl"), "wb") as f:
        pickle.dump(info, f)

    for step_idx, (step_bounds, relu_indexes) in enumerate(zip(branch_lp_bounds, relu_indexes_list)):
        step_folder = os.path.join(input_folder, f"test-step_{step_idx}")
        os.makedirs(step_folder, exist_ok=True)
        
        # Save relu_indexes as test-relu_nodes.json
        with open(os.path.join(step_folder, "test-relu_nodes.json"), "w") as f:
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
            
            
        with open(os.path.join(step_folder, "test-strong_branching.pkl"), "wb") as f:
            pickle.dump(branching_table, f)
            
        