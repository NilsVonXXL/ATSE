import json
import numpy as np

# Example path to relu_nodes.json
relu_json_path = 'c:/ATSE/ATSE/newDataset/blobs/data-blobs/2/input-x--0.1-y--0.1-eps-0.35/step_0/relu_nodes.json'

with open(relu_json_path, 'r') as f:
    relu_status = json.load(f)

print('ReLU status dictionary:', relu_status)
print(relu_status["2"])# Get sorted node indices
print(len(relu_status))

relu_status = np.array(list(relu_status.values()))
print('ReLU status array:', relu_status)

x = 0
target_node = -1
for idx, i in enumerate(relu_status):
    if i == 1:
        x += 1
    if (x == 2):
        target_node = idx
        break
    
print(x, target_node)

print(type(relu_status))