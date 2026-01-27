import subprocess
import time
import os

# Settings
num_sessions = 20
limit_per_session = 500
script_path = "storage/extraction_data_random.py"
parent_dir = "randomDataset(-5,5)"
os.makedirs(parent_dir, exist_ok=True)
base_dataset_dir = os.path.join(parent_dir, "randomDataset_parallel")



for i in range(num_sessions):
    session_name = f"extract_{i}"
    dataset_dir = f"{base_dataset_dir}_{i}"
    log_file = f"{dataset_dir}_log.txt"
    cmd = f"tmux new-session -d -s {session_name} 'python {script_path} --max-input-folders {limit_per_session} --dataset-dir {dataset_dir} > {log_file} 2>&1'"
    print(f"Starting tmux session: {session_name} -> {cmd}")
    subprocess.run(cmd, shell=True)
    time.sleep(0.2)  # Small delay to avoid race conditions

print("All tmux sessions started.")
