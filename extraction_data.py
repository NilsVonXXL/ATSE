import numpy as np
import pickle
from micrograd.engine import Value
from micrograd.ibp import Interval
from micrograd.branch_and_bound import branch_and_bound
import glob

def extract_network_data(model_path, input, eps):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    weights = np.array([p.data for p in model.parameters()])
    x = [Value(c) for c in input]
    in_bounds = {xi: Interval(xi.data - eps, xi.data + eps) for xi in x}
    score = model(x)
    best_lb, best_ub, minimizer, branch_lp_bounds = branch_and_bound(score, in_bounds)
    return {
        'weights': weights,
        'in_bounds': np.array([[b.lower, b.upper] for b in in_bounds.values()]),
        'best_lb': best_lb,
        'best_ub': best_ub,
        'minimizer': minimizer,
        'branch_lp_bounds': branch_lp_bounds,  # This is a list of lists of LP bounds per branch
    }
    
    
model_paths = glob.glob("models/*.pkl")  
input = [0.0, 0.0]  
eps = 0.1            

data_list = []
for model_path in model_paths:
    data = extract_network_data(model_path, input, eps)
    data_list.append(data)

np.save('gnn_training_data.npy', np.array(data_list, dtype=object))