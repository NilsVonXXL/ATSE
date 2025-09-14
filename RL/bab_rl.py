from dataclasses import dataclass
import itertools
import random
import numpy as np

from micrograd.engine import Value
from micrograd.ibp import ibp, Interval
from micrograd.planet import planet_relaxation
from micrograd.computegraph import rerun


def simple_deterministic(relu_nodes):
    return relu_nodes[0]


def collect_relu_nodes(output_node, node_bounds, splitted_nodes=()):
    """Collect mixed-phase ReLU nodes in a computation graph."""
    relu_nodes = []
    relu_indexes = []
    i = 0
    for v in output_node.compute_graph():
        if v.op == 'ReLU':
            v_inp = v.prev[0]
            if v_inp in splitted_nodes:
                # ReLU is already split, skip
                relu_indexes[i] = 0
                i += 1
                continue
            # Only add if lower and upper bounds have a sign change
            lower, upper = node_bounds[v_inp]
            if lower * upper < 0:
                relu_nodes.append(v)
                relu_indexes[i] = 1
                i += 1
                continue
            relu_indexes[i] = 0
            i += 1
            
    return relu_nodes, relu_indexes


@dataclass
class Branch:
    # the bounds for the *inputs* of the ReLUs that have been split
    splits: dict[Value, Interval]
    id: int
    depth: int = 0
    

def bab_step(score, in_bounds, splits, action):
    """
    Perform one BaB step for RL.
    score: output node
    in_bounds: input bounds
    splits: dict of already applied splits {Value: Interval}
    action: index of relu node to split (from splittable nodes)
    Returns: next_splits, relu_status, done, info
    """
    node_bounds = ibp(score, in_bounds, return_all=True)
    relu_nodes, relu_indexes = collect_relu_nodes(score, node_bounds, splits.keys())
    chosen_relu = relu_nodes[action]
    relu_input = chosen_relu.prev[0]
    relu_input_lb, relu_input_ub = node_bounds[relu_input]

    # Try both splits
    split1 = splits | {relu_input: Interval(relu_input_lb, 0.0)}
    split2 = splits | {relu_input: Interval(0.0, relu_input_ub)}

    # Evaluate split1
    try:
        branch_lb1, minimizer1 = planet_relaxation(score, in_bounds, node_bounds | split1)
    except Exception as e:
        print(f"planet_relaxation failed for split1: {e}")
        branch_lb1, minimizer1 = float('inf'), float('inf')
    branch_ub1 = rerun(score, minimizer1)

    # Evaluate split2
    try:
        branch_lb2, minimizer2 = planet_relaxation(score, in_bounds, node_bounds | split2)
    except Exception as e:
        print(f"planet_relaxation failed for split2: {e}")
        branch_lb2, minimizer2 = float('inf'), float('inf')
    branch_ub2 = rerun(score, minimizer2)

    # Decision logic
    # If both splits are prunable (lb >= 0), verification is complete
    if branch_lb1 >= 0 and branch_lb2 >= 0:
        done = True
        return splits, relu_indexes, done

    # If counterexample found in either split (ub < 0), verification is complete
    if branch_ub1 < 0 or branch_ub2 < 0:
        done = True
        return splits, relu_indexes, done

    # If one split is infeasible (lb == inf), choose the other one
    if branch_lb1 == float('inf') and branch_lb2 != float('inf'):
        chosen_split = split2
    elif branch_lb2 == float('inf') and branch_lb1 != float('inf'):
        chosen_split = split1
    # If both are infeasible
    #TODO: handle this case better
    elif branch_lb1 == float('inf') and branch_lb2 == float('inf'):
        done = True
        return splits, relu_indexes, done
    else:
        # If both are feasible and not pruned/counterexample, randomly choose one
        randI = random.random()
        chosen_split = split1 if randI > 0.5 else split2

    # Update relu_status: set chosen node as not splittable
    relu_indexes[action] = 0
    done = not np.any(relu_indexes == 1)

    return chosen_split, relu_indexes, done
