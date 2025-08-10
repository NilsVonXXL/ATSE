import argparse
import random
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

import numpy as np
np.random.seed(args.seed)
import random
random.seed(args.seed)

if args.dataset == 'moons':
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.1)
elif args.dataset == 'circles':
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=100, noise=0.1)
elif args.dataset == 'blobs':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, centers=2)
elif args.dataset == 'classification':
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
elif args.dataset == 'gaussian':
    from sklearn.datasets import make_gaussian_quantiles
    X, y = make_gaussian_quantiles(n_samples=100, n_features=2, n_classes=2)
else:
    raise ValueError("Unknown dataset")

y = y * 2 - 1

model = MLP(2, [16, 16, 1])

def loss(batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
        
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

total_loss, acc = loss()

for k in range(35):
    # forward
    total_loss, acc = loss()
    # backward
    model.zero_grad()
    total_loss.backward()
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, args.output)

with open(output_path, 'wb') as f:
    pickle.dump(model, f)