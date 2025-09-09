import os
import pickle
import json

# Set the root folder for test data
root = "test-data-moons/1"
print(f"Checking contents of: {root}")

# Print parameters
params_path = os.path.join(root, "test-parameters.pkl")
if os.path.exists(params_path):
    with open(params_path, "rb") as f:
        weights = pickle.load(f)
    for i in weights:
        if i.prev != ():
            print(i.data, i.prev);
            break
    print("No parameters with predecessors found.")
else:
    print("No parameters file found.")

# List input folders
input_folders = [d for d in os.listdir(root) if d.startswith("test-x-")]
print("Input folders:", input_folders)

for input_folder in input_folders:
    input_path = os.path.join(root, input_folder)
    info_path = os.path.join(input_path, "test-info.pkl")
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        print(f"\nInfo for {input_folder}:")
        print(info)
    else:
        print(f"No info.pkl in {input_folder}")

    # List branching steps
    step_folders = [d for d in os.listdir(input_path) if d.startswith("test-step_")]
    print("  Branching steps:", step_folders)
    for step_folder in step_folders:
        step_path = os.path.join(input_path, step_folder)
        relu_json = os.path.join(step_path, "test-relu_nodes.json")
        branching_pkl = os.path.join(step_path, "test-strong_branching.pkl")
        if os.path.exists(relu_json):
            with open(relu_json, "r") as f:
                relu_status = json.load(f)
            print(f"    {step_folder} relu status:", relu_status)
        if os.path.exists(branching_pkl):
            with open(branching_pkl, "rb") as f:
                branching_table = pickle.load(f)
            print(f"    {step_folder} branching table (all):", branching_table)

print("\nTest data structure printed successfully.")