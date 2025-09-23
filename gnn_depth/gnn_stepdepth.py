import torch
import pickle
import os
from step_depth.env import DeepThought42
from train_strong_branching_nn import FeedForwardNN
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "newDataset")
MODEL_PATH = os.path.join(BASE_DIR, "gnn_depth", "strong_branching_nn_eval.pt")

# Model setup
weights_dim = 337
relu_dim = 32
inputs_dim = 3
input_dim = weights_dim + relu_dim + inputs_dim

model = FeedForwardNN(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

stepdepth = []
initial_states = []

# Gather all initial states (same as RL)
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
            weights_path = os.path.join(net_path, 'parameters.pkl')
            if not os.path.exists(weights_path):
                continue
            input_folders = [f for f in os.listdir(net_path) if f.startswith('input-x-')]
            for input_folder in input_folders:
                input_path = os.path.join(net_path, input_folder)
                if os.path.isdir(input_path):
                    initial_states.append((weights_path, input_path, input_folder))

for w, i, f in tqdm(initial_states, desc="Evaluating GNN on instances"):
    env = DeepThought42(models_dir=MODELS_DIR, initial_states=(w, i, f))
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        features = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
            action = torch.argmax(output, dim=1).item()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    #print("For", w, i, f, "solved in", steps, "steps")
    stepdepth.append(steps)

with open("gnn_stepdepth.pkl", "wb") as f:
    pickle.dump(stepdepth, f)