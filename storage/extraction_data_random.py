import itertools
import numpy as np
import pickle
import os
import json
from micrograd.engine import Value
from micrograd.ibp import Interval
from micrograd.nn import MLP  # Assuming your random-initialized NN class is MLP
from micrograd.branch_and_bound import branch_and_bound
from tqdm import tqdm

def input_folder_name(x, y, eps):
    return f"input-x-{x}-y-{y}-eps-{eps}"

# Settings
NUM_MODELS = 10  # Number of random models to generate
DATASET_DIR = "randomDataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Input grid
x_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
y_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
eps_vals = np.round(np.arange(0.25, 0.41, 0.1), 2)
input_combinations = list(itertools.product(x_vals, y_vals, eps_vals))


for model_idx in tqdm(range(NUM_MODELS), desc="Random Models"):
    # Create random model
    model = MLP(2, [16, 16, 1])  # Init random NN (see MLP for random distribution)

    # Randomly sample 1/10 of input combinations
    num_samples = max(1, len(input_combinations) // 10)
    sampled_inputs = np.random.choice(len(input_combinations), size=num_samples, replace=False)

    # Check how many sampled points are in (0 <= score.data <= 2)
    skip_count = 0
    for idx in sampled_inputs:
        x, y, eps = input_combinations[idx]
        input_vals = [x, y]
        input_vars = [Value(val) for val in input_vals]
        score = model(input_vars)
        if (0 <= score.data <= 2):
            skip_count += 1
    
    if skip_count > num_samples * 0.25:
        continue
    
    #Only create folders if we are keeping this model
    model_folder = os.path.join(DATASET_DIR, f"random_{model_idx}")
    os.makedirs(model_folder, exist_ok=True)

    # Save model weights
    weights = [p for p in model.parameters()]
    with open(os.path.join(model_folder, "parameters.pkl"), "wb") as f:
        pickle.dump(weights, f)
    
    # Now, actually process the sampled points for this model
    for idx, (x, y ,eps) in enumerate(tqdm(input_combinations, desc=f"Model {model_idx} Inputs", leave=False)):
        input_vals = [x, y]
        input_vars = [Value(val) for val in input_vals]
        in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in input_vars}
        score = model(input_vars)
        if (0 <= score.data <= 2):
            continue

        # Run branch-and-bound
        best_lb, best_ub, minimizer, branch_lp_bounds, relu_indexes_list = branch_and_bound(score, in_bounds)

        # Only keep points with at least one branching step
        if not branch_lp_bounds:
            continue

        input_folder = os.path.join(model_folder, input_folder_name(x, y, eps))
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

            # Save relu_indexes as relu_nodes.json
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
