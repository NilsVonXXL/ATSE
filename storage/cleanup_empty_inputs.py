import os
import shutil

# Set the base directory where your randomDataset_parallel_* folders are
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
# Find all dataset dirs
for dataset_dir in os.listdir(base_dir):
    if not dataset_dir.startswith("randomDataset_parallel_"):
        print(dataset_dir)
        continue
    dataset_path = os.path.join(base_dir, dataset_dir)
    if not os.path.isdir(dataset_path):
        continue
    print(f"Checking {dataset_path}")
    # For each random_N folder
    for model_folder in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_folder)
        if not os.path.isdir(model_path):
            continue
        # For each input-x-y-eps-z folder
        for input_folder in os.listdir(model_path):
            input_path = os.path.join(model_path, input_folder)
            if not os.path.isdir(input_path):
                continue
            # Check for any step_* folder inside
            has_step = any(
                os.path.isdir(os.path.join(input_path, f)) and f.startswith("step_")
                for f in os.listdir(input_path)
            )
            if not has_step:
                print(f"Deleting {input_path} (no step_* folder)")
                shutil.rmtree(input_path)
        # After cleaning input folders, check if model folder is now empty (no input-x-y-eps-* folders)
        remaining_inputs = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and f.startswith("input-x-")]
        if not remaining_inputs:
            print(f"Deleting {model_path} (no input-x-y-eps-* folders left)")
            shutil.rmtree(model_path)
print("Cleanup complete.")
