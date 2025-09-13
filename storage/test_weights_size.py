import pickle
import numpy as np

# Path to a sample parameters.pkl file
weights_path = 'newDataset/blobs/data-blobs/4/parameters.pkl'

with open(weights_path, 'rb') as f:
    weights = pickle.load(f)

weights_flat = np.array([w.data for w in weights], dtype=np.float32)
print('Weights shape:', weights_flat.shape)
print('Weights length:', len(weights_flat))
print(weights_flat.dtype)
print(weights_flat[0])
