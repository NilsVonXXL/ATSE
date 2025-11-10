import os
import pickle
import matplotlib.pyplot as plt
import sys
# --- 1. Count step depths from folders ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "newDataset")
CLASS_DIR = os.path.join(BASE_DIR, "classDataset")
folder_stepdepths = []

# Collect (step_depth, input_folder_path) pairs
folder_depth_pairs = []

for domain in os.listdir(DATASET_DIR):
    domain_path = os.path.join(DATASET_DIR, domain)
    if not os.path.isdir(domain_path):
        continue
    for data_subdir in os.listdir(domain_path):
        data_path = os.path.join(domain_path, data_subdir)
        if not os.path.isdir(data_path):
            continue
        for net_num in os.listdir(data_path):
            net_path = os.path.join(data_path, net_num)
            if not os.path.isdir(net_path):
                continue
            input_folders = [f for f in os.listdir(net_path) if f.startswith('input-x-')]
            for f in input_folders:
                input_folder_path = os.path.join(net_path, f)
                if os.path.isdir(input_folder_path):
                    n = len(os.listdir(input_folder_path))
                    folder_stepdepths.append(n - 1)
                    folder_depth_pairs.append((n - 1, input_folder_path))


# --- 2. Load pickle files ---
with open(os.path.join("gnn_stepdepth.pkl"), 'rb') as f:
    gnn_stepdepths = pickle.load(f)
with open(os.path.join("stepdepth.pkl"), 'rb') as f:
    rl_stepdepth = pickle.load(f)
with open(os.path.join("gnn_stepdepthC.pkl"), 'rb') as f:
    gnn_stepdepths_alt = pickle.load(f)
with open(os.path.join("stepdepthC.pkl"), 'rb') as f:
    rl_stepdepth_alt = pickle.load(f)


# --- Count step depths for input folders in 'classDataset' folder ---
classification_stepdepths = []
for domain in os.listdir(CLASS_DIR):
    domain_path = os.path.join(CLASS_DIR, domain)
    if not os.path.isdir(domain_path):
        continue
    for data_subdir in os.listdir(domain_path):
        data_path = os.path.join(domain_path, data_subdir)
        if not os.path.isdir(data_path):
            continue
        for net_num in os.listdir(data_path):
            net_path = os.path.join(data_path, net_num)
            if not os.path.isdir(net_path):
                continue
            # Directly look for input-x-* folders here
            input_folders = [f for f in os.listdir(net_path) if f.startswith('input-x-')]
            for f in input_folders:
                input_folder_path = os.path.join(net_path, f)
                if os.path.isdir(input_folder_path):
                    n = len(os.listdir(input_folder_path))
                    classification_stepdepths.append(n - 1)

# --- Plot boxplots (only once, at the end) ---
plt.figure(figsize=(20, 6))
labels = [
    'SB', 'SB (test)',
    'GNN', 'GNN (test)',
    'RL', 'RL (test)'
    
]
data = [
    folder_stepdepths, classification_stepdepths,
    gnn_stepdepths, gnn_stepdepths_alt,
    rl_stepdepth, rl_stepdepth_alt
    
]

plt.subplot(1, 2, 1)
plt.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    notch=True,
    showmeans=True,
    meanline=True
)
plt.title('Full Range')

plt.subplot(1, 2, 2)
plt.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    notch=True,
    showmeans=True,
    meanline=True
)
plt.ylim(0, 6)
plt.title('Zoomed In (0-6)')
plt.tight_layout()
plt.show()