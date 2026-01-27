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
MAX_INPUT_FOLDERS = 2000  # Set your desired limit here
DATASET_DIR = "randomDataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Input grid
x_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
y_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
eps_vals = np.round(np.arange(0.25, 0.41, 0.1), 2)
input_combinations = list(itertools.product(x_vals, y_vals, eps_vals))

input_folder_count = 0
model_idx = 0

with tqdm(total=MAX_INPUT_FOLDERS, desc="Input Folders Created") as pbar:
    while input_folder_count < MAX_INPUT_FOLDERS:
        # Create random model
        model = MLP(2, [16, 16, 1])  # Init random NN (see MLP for random distribution)

        # Save model weights
        model_folder = os.path.join(DATASET_DIR, f"random_{model_idx}")
        os.makedirs(model_folder, exist_ok=True)
        weights = [p for p in model.parameters()]
        with open(os.path.join(model_folder, "parameters.pkl"), "wb") as f:
            pickle.dump(weights, f)

        # Shuffle input combinations for this model
        np.random.shuffle(input_combinations)

        for idx, (x, y, eps) in enumerate(input_combinations):
            if input_folder_count >= MAX_INPUT_FOLDERS:
                break
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

            input_folder_count += 1
            pbar.update(1)

        model_idx += 1
