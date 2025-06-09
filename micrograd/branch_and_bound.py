import random
from micrograd.ibp import ibp, Interval
from micrograd.engine import Value
from micrograd.helpers import collect_relu_nodes, planet_relaxation
import copy


def branch_and_bound(score, in_bounds, splits={}):
    
    # ibp
    node_bounds = ibp(score, in_bounds | splits, return_all=True)
    
    
    
    # Collect ReLU nodes
    relu_nodes = collect_relu_nodes(score, node_bounds, splits.keys())
    #print("test")
    # No ReLU nodes with sign change in bounds found
    if not relu_nodes:
        interval = node_bounds[score]
        return interval.lower, interval.upper
    
    #chosen_relu = random.choice(relu_nodes)
    chosen_relu = relu_nodes[0]  # For deterministic behavior, use the first one
    relu_input = list(chosen_relu.prev)[0]
    print(relu_nodes)
    
    # Branch 1: ReLU input >= 0
    
    split1 = copy.copy(splits)
    split1[relu_input] = Interval(0, node_bounds[relu_input].upper)
    
    # Check if the bounds are valid for the first branch via Planet relaxation
    check1_l, check1_u = planet_relaxation(score, in_bounds ,node_bounds | split1)
    if check1_l == float('inf') or check1_u == float('-inf'):
        bounds1 =  float('inf'), float('-inf')
    elif check1_l >= 0:
        bounds1= check1_l, check1_u
    elif check1_u < 0:
        return check1_l, check1_u
    else:
        bounds1 = branch_and_bound(score, in_bounds, split1)

    # Branch 2: ReLU input <= 0
    
    split2 = copy.copy(splits)
    split2[relu_input] = Interval(node_bounds[relu_input].lower, 0)
    
    check2_l, check2_u = planet_relaxation(score, in_bounds, node_bounds | split2)
    if check2_l == float('inf') or check2_u == float('-inf'):
        bounds2 = float('inf'), float('-inf')
    elif check2_l >= 0:
        bounds2= check2_l, check2_u
    elif check2_u < 0:
        return check2_l, check2_u
    else:
        bounds2 = branch_and_bound(score, in_bounds, split2)
    

    # Return global bounds
    print (f"bounds1: {bounds1}, bounds2: {bounds2}")  # debugging
    return min(bounds1[0], bounds2[0]), max(bounds1[1], bounds2[1])
   