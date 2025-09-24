import os
import pickle
import matplotlib.pyplot as plt
import sys
# --- 1. Count step depths from folders ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "newDataset")
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

# Sort and print the 10 largest step depths with their folder paths
#top_10 = sorted(folder_depth_pairs, key=lambda x: x[0], reverse=True)[:10]
#print("\nTop 10 longest step depths and their input folders:")
#for depth, path in top_10:
#    print(f"Step depth: {depth}, Folder: {path}")

# --- 2. Load pickle files ---
with open(os.path.join("gnn_stepdepth.pkl"), 'rb') as f:
    gnn_stepdepths = pickle.load(f)

with open(os.path.join("stepdepth.pkl"), 'rb') as f:
    rl_stepdepth = pickle.load(f)


print(f"Dataset step depths: {folder_stepdepths.__len__()}")
print(f"GNN step depths: {gnn_stepdepths.__len__()}")
print(f"RL step depths: {rl_stepdepth.__len__()}")

print(f"Dataset avg step depth: {sum(folder_stepdepths)/len(folder_stepdepths):.2f}")
print(f"GNN avg step depth: {sum(gnn_stepdepths)/len(gnn_stepdepths):.2f}")
print(f"RL avg step depth: {sum(rl_stepdepth)/len(rl_stepdepth):.2f}")

#sys.exit()

# --- 3. Plot boxplots ---
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.boxplot(
    [folder_stepdepths, gnn_stepdepths, rl_stepdepth],
    labels=['Dataset', 'GNN', 'RL'],
    patch_artist=True,
    notch=True,
    showmeans=True,
    meanline=True
)
plt.title('Full Range')

plt.subplot(1, 2, 2)
plt.boxplot(
    [folder_stepdepths, gnn_stepdepths, rl_stepdepth],
    labels=['Strong Branching', 'GNN', 'RL'],
    patch_artist=True,
    notch=True,
    showmeans=True,
    meanline=True
)
plt.ylim(0, 6)
plt.title('Zoomed In (0-6)')
plt.tight_layout()
plt.show()