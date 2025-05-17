import random
import numpy as np
import matplotlib.pyplot as plt

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

np.random.seed(1337)
random.seed(1337)

# make up a dataset

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

# Get the first data point and label
x0, y0 = X[0], y[0]

# initialize a model 
model = MLP(2, [16, 16, 1]) # 2-layer neural network


# loss function
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

# Uncomment below to see IBP bounds before training
print("\n--- IBP Analysis on untrained network ---")
xrow = X[0]
input_with_bounds = [Value(x) for x in xrow]
eps = 0.1
#input consits of 2 vlaues
input_with_bounds[0].lower = input_with_bounds[0].data - eps
input_with_bounds[0].upper = input_with_bounds[0].data + eps

input_with_bounds[1].lower = input_with_bounds[1].data - eps
input_with_bounds[1].upper = input_with_bounds[1].data + eps
score = model(input_with_bounds)
score.ibp()
print(f"Output bounds for input 0 (untrained): lower={score.lower}, upper={score.upper}")

# optimization
for k in range(100):
    # forward
    total_loss, acc = loss()
    # backward
    model.zero_grad()
    total_loss.backward()
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

# After training: analyze bounds for a specific input
print("\n--- IBP Analysis on trained network ---")
xrow = X[0] 
input_with_bounds = [Value(x) for x in xrow]
eps = 0.1
#input consits of 2 vlaues
input_with_bounds[0].lower = input_with_bounds[0].data - eps
input_with_bounds[0].upper = input_with_bounds[0].data + eps

input_with_bounds[1].lower = input_with_bounds[1].data - eps
input_with_bounds[1].upper = input_with_bounds[1].data + eps

score = model(input_with_bounds)
score.ibp()
print(f"Output bounds for input 0: lower={score.lower}, upper={score.upper}")

# Custom input for IBP analysis
#print("\n--- IBP Analysis on custom input ---")

#custom_x0 = 1  
#custom_x1 = 0 
#custom_input = [Value(custom_x0), Value(custom_x1)]
#eps = 0.1
#custom_input[0].lower = custom_input[0].data - eps
#custom_input[0].upper = custom_input[0].data + eps
#custom_input[1].lower = custom_input[1].data - eps
#custom_input[1].upper = custom_input[1].data + eps
#score = model(custom_input)
#score.ibp()
#print(f"Output bounds for custom input: lower={score.lower}, upper={score.upper}")


