import random
from micrograd.ibp import ibp, Interval
from micrograd.engine import Value
from micrograd.helpers import collect_relu_nodes, planet_relaxation
import copy


def branch_and_bound(score, in_bounds, splits={}, node_bounds=None):
    
    # ibp
    if node_bounds is None:
        node_bounds = ibp(score, in_bounds, return_all=True)
    
    
    
    # Collect ReLU nodes
    relu_nodes = collect_relu_nodes(score, node_bounds, splits.keys())
    
    # No ReLU nodes with sign change in bounds found
    if not relu_nodes:
        print("No ReLU nodes with sign change found, returning bounds")
        return planet_relaxation(score, in_bounds, node_bounds)
    
    #chosen_relu = random.choice(relu_nodes)
    chosen_relu = relu_nodes[0]  # For deterministic behavior, use the first one
    relu_input = list(chosen_relu.prev)[0]
    print("Ammout of Relus", len(relu_nodes))
    
    # Branch 1: ReLU input >= 0
    
    split1 = copy.copy(splits)
    split1[relu_input] = Interval(0, node_bounds[relu_input].upper)
    
    # Check if the bounds are valid for the first branch via Planet relaxation
    check1_l, check1_u = planet_relaxation(score, in_bounds ,node_bounds | split1)
    if check1_l == float('inf') or check1_u == float('-inf'):
        print("Branch 1 is infeasible, bounds:", check1_l, check1_u)
        bounds1 =  float('inf'), float('-inf')
    elif check1_l >= 0:
        print("Branch 1 is satfisfied, bounds:", check1_l, check1_u)
        bounds1= check1_l, check1_u
    elif check1_u < 0:
        print("Counterexample found in branch 1")
        return check1_l, check1_u
    else:
        print("Continuing branch and bound for branch 1", check1_l, check1_u)
        # counter implematation for branching how deep
        bounds1 = branch_and_bound(score, in_bounds, split1, node_bounds)

    # Branch 2: ReLU input <= 0
    
    split2 = copy.copy(splits)
    split2[relu_input] = Interval(node_bounds[relu_input].lower, 0)
    
    check2_l, check2_u = planet_relaxation(score, in_bounds, node_bounds | split2)
    if check2_l == float('inf') or check2_u == float('-inf'):
        print("Branch 2 is infeasible, bounds:", check2_l, check2_u)
        bounds2 = float('inf'), float('-inf')
    elif check2_l >= 0:
        print("Branch 2 is satfisfied, bounds:", check2_l, check2_u)
        bounds2= check2_l, check2_u
    elif check2_u < 0:
        print("Counterexample found in branch 2")
        return check2_l, check2_u
    else:
        print("Continuing branch and bound for branch 2", check2_l, check2_u)
        bounds2 = branch_and_bound(score, in_bounds, split2, node_bounds)
    

    # Return global bounds
    #print (f"bounds1: {bounds1}, bounds2: {bounds2}")  # debugging
    print("End boudns", bounds1, bounds2)
    return min(bounds1[0], bounds2[0]), max(bounds1[1], bounds2[1])
   