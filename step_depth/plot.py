import os
import pickle
import matplotlib.pyplot as plt
import sys
# --- 1. Count step depths from folders ---

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "oldDataset")
folder_stepdepths = []

# Collect (step_depth, input_folder_path) pairs (only for oldDataset)
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



# --- 2. Load only two pickle files (GNN and RL) ---
with open(os.path.join("gnn_stepdepth_0802.pkl"), 'rb') as f:
    gnn_stepdepths = pickle.load(f)
with open(os.path.join("stepdepth_0802_nD.pkl"), 'rb') as f:
    rl_stepdepth_nD = pickle.load(f)
with open(os.path.join("stepdepth_0802_(-1,1).pkl"), 'rb') as f:
    rl_stepdepth_neg11 = pickle.load(f)
with open(os.path.join("stepdepth_0802_(gauss).pkl"), 'rb') as f:
    rl_stepdepth_gauss = pickle.load(f)


# --- Plot boxplots for only three datasets ---
plt.figure(figsize=(12, 6))
labels = [
    'SB',
    'GNN',
    'RL(nD)',
    'RL(-1,1)',
    'RL(gauss)'
]
data = [
    folder_stepdepths,
    gnn_stepdepths,
    rl_stepdepth_nD,
    rl_stepdepth_neg11,
    rl_stepdepth_gauss
]

plt.subplot(1, 2, 1)
plt.boxplot(
    data,
    tick_labels=labels,
    patch_artist=True,
    notch=True,
    showmeans=True,
    meanline=True
)
plt.title('Full Range')

plt.subplot(1, 2, 2)
plt.boxplot(
    data,
    tick_labels=labels,
    patch_artist=True,
    notch=True,
    showmeans=True,
    meanline=True
)
plt.ylim(0, 6)
plt.title('Zoomed In (0-6)')
plt.tight_layout()
plt.show()