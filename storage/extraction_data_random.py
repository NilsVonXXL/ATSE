import itertools
import numpy as np
import pickle
import os
import json
import argparse
from micrograd.engine import Value
from micrograd.ibp import Interval
from micrograd.nn import MLP  # Assuming your random-initialized NN class is MLP
from micrograd.branch_and_bound import branch_and_bound
from tqdm import tqdm

def input_folder_name(x, y, eps):
    return f"input-x-{x}-y-{y}-eps-{eps}"


def main():
    parser = argparse.ArgumentParser(description="Generate random dataset with random-initialized NNs and branch-and-bound steps.")
    parser.add_argument('--max-input-folders', type=int, default=2000, help='Maximum number of input folders to create (default: 2000)')
    parser.add_argument('--dataset-dir', type=str, default='randomDataset', help='Directory to store the generated dataset (default: randomDataset)')
    args = parser.parse_args()

    MAX_INPUT_FOLDERS = args.max_input_folders
    DATASET_DIR = args.dataset_dir
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Input grid
    x_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
    y_vals = np.round(np.arange(-2.0, 2.51, 0.1), 2)
    eps_vals = np.round(np.arange(0.05, 0.31, 0.1), 2)
    input_combinations = list(itertools.product(x_vals, y_vals, eps_vals))

    input_folder_count = 0
    model_idx = 0

    with tqdm(total=MAX_INPUT_FOLDERS, desc="Input Folders Created") as pbar:
        while input_folder_count < MAX_INPUT_FOLDERS:
            # Create random model
            model = MLP(2, [16, 16, 1])  # Init random NN (see MLP for random distribution)

            # Shuffle input combinations for this model
            np.random.shuffle(input_combinations)

            model_folder = None
            weights = None
            model_has_input = False

            for _, (x, y, eps) in enumerate(input_combinations):
                if input_folder_count >= MAX_INPUT_FOLDERS:
                    break
                try:
                    input_vals = [x, y]
                    input_vars = [Value(val) for val in input_vals]
                    in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in input_vars}
                    score = model(input_vars)
                    if (-2 <= score.data <= 2):
                        continue

                    # Run branch-and-bound
                    best_lb, best_ub, minimizer, branch_lp_bounds, relu_indexes_list = branch_and_bound(score, in_bounds)
                    if not branch_lp_bounds:
                        continue

                    # Only create model folder and save weights if this is the first valid input
                    if not model_has_input:
                        model_folder = os.path.join(DATASET_DIR, f"random_{model_idx}")
                        os.makedirs(model_folder, exist_ok=True)
                        weights = [p for p in model.parameters()]
                        with open(os.path.join(model_folder, "parameters.pkl"), "wb") as f:
                            pickle.dump(weights, f)
                        model_has_input = True

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
                except Exception as e:
                    print(f"[ERROR] Exception for input ({x}, {y}, {eps}): {e}")

            model_idx += 1

if __name__ == "__main__":
    main()
