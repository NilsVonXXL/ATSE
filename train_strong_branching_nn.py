import json
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

# --- CONFIG ---
DATA_DIR = 'newDataset'  # Use newDataset as source
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

# --- DATASET ---
class StrongBranchingDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for domain in os.listdir(root_dir):
            domain_path = os.path.join(root_dir, domain)
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
                    # Load weights
                    weights_path = os.path.join(net_path, 'parameters.pkl')
                    if not os.path.exists(weights_path):
                        continue
                    with open(weights_path, 'rb') as f:
                        weights = pickle.load(f)
                    # For each input folder
                    for input_folder in os.listdir(net_path):
                        input_path = os.path.join(net_path, input_folder)
                        if not os.path.isdir(input_path):
                            continue
                        # Parse input values from folder name
                        try:
                            match = re.match(r'input-x-([-\d.]+)-y-([-\d.]+)-eps-([-\d.]+)', input_folder)
                            if match:
                                x = float(match.group(1))
                                y = float(match.group(2))
                                eps = float(match.group(3))
                                inputs = [x, y, eps]
                            else:
                                print('Could not parse input folder name!')
                        except Exception: 
                            continue
                        # For each step
                        for step_folder in os.listdir(input_path):
                            step_path = os.path.join(input_path, step_folder)
                            if not os.path.isdir(step_path):
                                continue
                            relu_json_path = os.path.join(step_path, 'relu_nodes.json')
                            branching_pkl_path = os.path.join(step_path, 'strong_branching.pkl')
                            if not (os.path.exists(relu_json_path) and os.path.exists(branching_pkl_path)):
                                continue
                            with open(relu_json_path, 'r') as f:
                                relu_status = json.load(f)
                           
                            with open(branching_pkl_path, 'rb') as f:
                                branching_tables = pickle.load(f)
                            # For each entry in the branching table, create a sample
                            for entry in branching_tables:
                                score_val = min(entry['split_left_lb'], entry['split_right_lb'])
                                if score_val > best_score:
                                    best_score = score_val
                                    target_index = entry['relu_index']
                                    
                            x = 0
                            for i in relu_status:
                                if i == 1:
                                    x += 1
                                else:
                                    continue
                                if x == target_index:
                                    target_node = i
                                    break
                             
                            
                            sample = {
                                'weights': np.array(weights).flatten(),
                                'inputs': np.array(inputs),
                                'relu_status': relu_status,
                                'target_node': target_node
                            }
                            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = np.concatenate([sample['weights'], sample['inputs'], sample['relu_status']])
        target = sample['target_node']
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

# --- MODEL ---
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, x):
        return self.net(x)


def main():
    dataset = StrongBranchingDataset(DATA_DIR)
    if len(dataset) == 0:
        print('No data found!')
        return
    weights_dim = 337
    relu_dim = 32
    inputs_dim = 3
    input_dim = weights_dim + relu_dim + inputs_dim
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = FeedForwardNN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for features, target in loader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), 'strong_branching_nn.pt')
    print('Model saved to strong_branching_nn.pt')

if __name__ == '__main__':
    main()
