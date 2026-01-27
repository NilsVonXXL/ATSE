import pickle
import numpy as np
from micrograd.nn import MLP

# Create a model and dump it
model1 = MLP(2, [16, 16, 1])
with open("test_model.pkl", "wb") as f:
    pickle.dump(model1, f)

# Save parameters only
params = [p for p in model1.parameters()]
with open("test_params.pkl", "wb") as f:
    pickle.dump(params, f)

# Reconstruct model from parameters
model2 = MLP(2, [16, 16, 1])
with open("test_params.pkl", "rb") as f:
    loaded_params = pickle.load(f)
for p_model, p_loaded in zip(model2.parameters(), loaded_params):
    p_model.data = p_loaded.data

# Compare models: check all parameter values
param_diffs = [abs(p1.data - p2.data) for p1, p2 in zip(model1.parameters(), model2.parameters())]
print(f"Max param diff: {np.max(param_diffs):.8f}")
print(f"Mean param diff: {np.mean(param_diffs):.8f}")

# Optionally, compare model outputs on random input
x = [np.random.randn(), np.random.randn()]
from micrograd.engine import Value
input_vals = [Value(val) for val in x]
out1 = model1(input_vals)
out2 = model2(input_vals)
print(f"Output diff: {abs(out1.data - out2.data):.8f}")
