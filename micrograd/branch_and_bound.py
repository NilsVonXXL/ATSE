from dataclasses import dataclass
import itertools

from micrograd.engine import Value
from micrograd.ibp import ibp, Interval
from micrograd.planet import planet_relaxation


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


@dataclass
class Branch:
    # the bounds for the *inputs* of the ReLUs that have been split
    splits: dict[Value, Interval]
    id: int
    depth: int = 0


def branch_and_bound(score, in_bounds, choose_relu=simple_deterministic):
    node_bounds = ibp(score, in_bounds, return_all=True)

    ids = itertools.count(0)
    branches = [Branch(splits={}, id=next(ids))]
    best_lb, best_ub = node_bounds[score]
    while len(branches) > 0:
        branch = branches.pop(0)

        branch_lb, branch_ub, minimizer = planet_relaxation(score, in_bounds, node_bounds | branch.splits)

        best_lb = min(best_lb, branch_lb)
        best_ub = min(best_ub, branch_ub)  # we search for bounds on the minimum of the score!

        if branch_lb == float('inf') or branch_ub == float('-inf'):
            print(f"Pruning infeasible branch {branch.id}.")
            continue
        elif branch_lb >= 0:
            print(f"Pruning satisfied branch {branch.id} with bounds: {branch_lb}, {branch_ub}")
            continue
        elif branch_ub < 0:
            print(f"Counterexample found in branch {branch.id} with bounds: {branch_lb}, {branch_ub}")
            return branch_lb, branch_ub, minimizer
        
        print(f"Splitting branch {branch.id} with bounds {branch_lb}, {branch_ub}")
        relu_nodes = collect_relu_nodes(score, node_bounds, branch.splits.keys())
        if not relu_nodes:
            print("All ReLU nodes split.")
            # in this case, the planet LP relaxation is precise and the minimizer
            # for the lower bound is a counterexample
            return best_lb, best_ub, minimizer
        
        chosen_relu = choose_relu(relu_nodes)
        relu_input = chosen_relu.prev[0]
        relu_input_lb, relu_input_ub = node_bounds[relu_input]

        split1 = branch.splits | {relu_input: Interval(relu_input_lb, 0.0)}
        split2 = branch.splits | {relu_input: Interval(0.0, relu_input_ub)}
        branches.append(Branch(splits=split1, id=next(ids), depth=branch.depth + 1))
        branches.append(Branch(splits=split2, id=next(ids), depth=branch.depth + 1))

    print("All branches pruned.")
    return best_lb, best_ub, None 
   