import pickle
import sys
import os
from micrograd.nn import MLP

def print_model_params(model_path):
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    print(f"Loaded {len(params)} parameters from {model_path}")
    values = []
    for i, p in enumerate(params):
        if hasattr(p, 'data'):
            print(f"Param {i}: value={p.data:.4f}, grad={getattr(p, 'grad', None)}")
            values.append(p.data)
        else:
            print(f"Param {i}: (type {type(p)}) - {p}")
    if values:
        import numpy as np
        mean = np.mean(values)
        std = np.std(values)
        print(f"\nParameter value distribution: mean={mean:.4f}, std={std:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_params.py <parameters.pkl>")
        sys.exit(1)
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)
    print_model_params(model_path)
