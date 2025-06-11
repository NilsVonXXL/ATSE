from micrograd.ibp import ibp, Interval
from micrograd.planet import planet_relaxation
import copy


def simple_deterministic(relu_nodes):
    return relu_nodes[0]


def collect_relu_nodes(output_node, node_bounds, splitted_nodes=()):
    """Collect mixed-phase ReLU nodes in a computation graph."""
    relu_nodes = []
    for v in output_node.compute_graph():
        if v.op == 'ReLU':
            v_inp = v.prev[0]
            if v_inp in splitted_nodes:
                # ReLU is already split, skip
                continue
            # Only add if lower and upper bounds have a sign change
            lower, upper = node_bounds[v_inp]
            if lower * upper < 0:
                relu_nodes.append(v)
    return relu_nodes


def branch_and_bound(score, in_bounds, choose_relu=simple_deterministic, splits={}, node_bounds=None):
    # ibp
    if node_bounds is None:
        node_bounds = ibp(score, in_bounds, return_all=True)
    
    # Collect ReLU nodes
    relu_nodes = collect_relu_nodes(score, node_bounds, splits.keys())
    
    # No ReLU nodes with sign change in bounds found
    if not relu_nodes:
        print("No ReLU nodes with sign change found, returning bounds")
        return planet_relaxation(score, in_bounds, node_bounds)
    
    chosen_relu = choose_relu(relu_nodes)
    relu_input = chosen_relu.prev[0]
    print("Number of Unstable Relus:", len(relu_nodes))
    
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
        bounds1 = branch_and_bound(score, in_bounds, choose_relu, split1, node_bounds)

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
        bounds2 = branch_and_bound(score, in_bounds, choose_relu, split2, node_bounds)
    

    # Return global bounds
    #print (f"bounds1: {bounds1}, bounds2: {bounds2}")  # debugging
    print("End boudns", bounds1, bounds2)
    return min(bounds1[0], bounds2[0]), max(bounds1[1], bounds2[1])
   