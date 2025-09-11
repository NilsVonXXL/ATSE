import pickle
import numpy as np

# Path to a sample parameters.pkl file
weights_path = 'newDataset/blobs/data-blobs/4/parameters.pkl'

with open(weights_path, 'rb') as f:
    weights = pickle.load(f)

weights_flat = np.array(weights).flatten()
print('Weights shape:', weights_flat.shape)
print('Weights length:', len(weights_flat))
