import pickle
import numpy as np
import os 


DATASET_PICKLE = 'strong_branching_samples.pkl'
with open(DATASET_PICKLE, 'rb') as f:
        samples = pickle.load(f)

i = 1
for sample in samples:
    if sample['target_node'] == -1:
        print(i)  # Skip invalid samples
        i += 1