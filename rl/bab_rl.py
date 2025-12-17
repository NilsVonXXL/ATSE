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
    relu_indexes = {}
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
    splits: dict  # {Value: Interval} constraints for this branch
    node_bounds: dict  # {Value: Interval} bounds for all nodes in this branch
    relu_indexes: dict  # {int: int} status of ReLUs (active/inactive) for observation
    


def bab_step(score, in_bounds, branch: Branch, action):
    """
    Expand a branch at the given action (ReLU index).
    Returns a list of valid child Branch objects (0, 1, or 2).
    Each Branch contains splits, node_bounds, relu_indexes.
    """
    splits = branch.splits
    node_bounds = branch.node_bounds
    relu_indexes = branch.relu_indexes.copy()

    relu_nodes, relu_status = collect_relu_nodes(score, node_bounds, splits.keys())
    # Check if action is valid
    if relu_status[action] == 0:
        # Invalid action: return empty list (caller can penalize)
        return [], None
    
    # Find the actual ReLU node to split
    c = -1
    for i, _ in enumerate(relu_status):
        if relu_status[i] == 1:
            c += 1
        if i == action:
            break
    chosen_relu = relu_nodes[c]
    relu_input = chosen_relu.prev[0]
    relu_input_lb, relu_input_ub = node_bounds[relu_input]

    # Try both splits
    split1 = splits | {relu_input: Interval(relu_input_lb, 0.0)}
    split2 = splits | {relu_input: Interval(0.0, relu_input_ub)}

    children = []
    done = False

    # Evaluate split1
    try:
        branch_lb1, minimizer1 = planet_relaxation(score, in_bounds, node_bounds | split1)
    except Exception as e:
        print(f"planet_relaxation failed for split1: {e}")
        branch_lb1, minimizer1 = float('inf'), None
    if branch_lb1 != float('inf'):
        try:
            branch_ub1 = rerun(score, minimizer1)
            print(f"branch_ub1: {branch_ub1}")  # Debugging
        except Exception as e:
            print(f"rerun failed for split1: {e}")
            branch_ub1 = float('inf')
    else:
        branch_ub1 = float('inf')

    # Only add child if not pruned
    if branch_lb1 != float('inf') and branch_lb1 < 0 and branch_ub1 >= 0:
        new_relu_indexes = relu_indexes.copy()
        new_relu_indexes[action] = 0
        new_node_bounds = ibp(score, in_bounds, return_all=True)
        children.append(Branch(splits=split1, node_bounds=new_node_bounds, relu_indexes=new_relu_indexes))
        done = False

    # Evaluate split2
    try:
        branch_lb2, minimizer2 = planet_relaxation(score, in_bounds, node_bounds | split2)
    except Exception as e:
        print(f"planet_relaxation failed for split2: {e}")
        branch_lb2, minimizer2 = float('inf'), None
    if branch_lb2 != float('inf'):
        try:
            branch_ub2 = rerun(score, minimizer2)
            print(f"branch_ub2: {branch_ub2}")  # Debugging
        except Exception as e:
            print(f"rerun failed for split2: {e}")
            branch_ub2 = float('inf')
    else:
        branch_ub2 = float('inf')

    if branch_lb2 != float('inf') and branch_lb2 < 0 and branch_ub2 >= 0:
        new_relu_indexes = relu_indexes.copy()
        new_relu_indexes[action] = 0
        new_node_bounds = ibp(score, in_bounds, return_all=True)
        children.append(Branch(splits=split2, node_bounds=new_node_bounds, relu_indexes=new_relu_indexes))
        done = False

    # Counterexample
    if branch_ub1 < 0 or branch_ub2 < 0:
        done = True

    return children, done
