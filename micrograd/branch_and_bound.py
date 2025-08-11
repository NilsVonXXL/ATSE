from dataclasses import dataclass
import itertools

from micrograd.engine import Value
from micrograd.ibp import ibp, Interval
from micrograd.planet import planet_relaxation
from micrograd.computegraph import rerun


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
    
    

# strong branching heurisitc
def strong_branching(relu_nodes, score, in_bounds, node_bounds, current_splits):
    best_node = None
    best_score = float('-inf')
    all_bounds = []

    for node in relu_nodes:
        relu_input = node.prev[0]
        relu_input_lb, relu_input_ub = node_bounds[relu_input]

        # Split 1: input <= 0
        split1 = current_splits | {relu_input: Interval(relu_input_lb, 0.0)}
        lb1, _ = planet_relaxation(score, in_bounds, node_bounds | split1)

        # Split 2: input >= 0
        split2 = current_splits | {relu_input: Interval(0.0, relu_input_ub)}
        lb2, _ = planet_relaxation(score, in_bounds, node_bounds | split2)

        all_bounds.append({
            'relu_node': node,
            'split1_lb': lb1,
            'split2_lb': lb2,
            'split1_bounds': (relu_input_lb, 0.0),
            'split2_bounds': (0.0, relu_input_ub)
        })

        score_val = min(lb1, lb2) 

        if score_val > best_score:
            best_score = score_val
            best_node = node

    return best_node, all_bounds
    
    
#FIXME: Watch out for new "all_bounds" return of strong branching

  
def branch_and_bound(score, in_bounds):
    branch_counter = 0
    branch_lp_bounds = []
    
    node_bounds = ibp(score, in_bounds, return_all=True)

    ids = itertools.count(0)
    _, best_ub = node_bounds[score]
    pruned_lbs = []  # used to compute the best lower bound
    branches = [Branch(splits={}, id=next(ids))]
    while len(branches) > 0:
        branch = branches.pop(0)

        branch_lb, minimizer = planet_relaxation(score, in_bounds, node_bounds | branch.splits)

        if branch_lb == float('inf'):
            print(f"Pruning infeasible branch {branch.id}.")
            continue

        branch_ub = rerun(score, minimizer)

        if branch_lb >= 0:
            print(f"Pruning satisfied branch {branch.id} with bounds: {branch_lb}, {branch_ub}")
            pruned_lbs.append(branch_lb)
            continue
        elif branch_ub < 0:
            print(f"Counterexample found in branch {branch.id} with bounds: {branch_lb}, {branch_ub}")
            return branch_lb, branch_ub, minimizer #FIXME: invalid lower bound(not min lower bound)

        best_ub = min(best_ub, branch_ub)  # we search for bounds on the minimum of the score!
        
        print(f"Splitting branch {branch.id} with bounds {branch_lb}, {branch_ub}")
        relu_nodes = collect_relu_nodes(score, node_bounds, branch.splits.keys())
        # If no ReLU nodes remain, the LP encoding is exact and we should have already exited
        # this branch above.
        assert relu_nodes is not None
        
        #choosing with strong branching
        chosen_relu, all_bounds = strong_branching(relu_nodes, score, in_bounds, node_bounds, branch.splits)
        branch_lp_bounds.append(all_bounds)
        
        
        branch_counter += 1
        
        relu_input = chosen_relu.prev[0]
        relu_input_lb, relu_input_ub = node_bounds[relu_input]

        split1 = branch.splits | {relu_input: Interval(relu_input_lb, 0.0)}
        split2 = branch.splits | {relu_input: Interval(0.0, relu_input_ub)}
        branches.append(Branch(splits=split1, id=next(ids), depth=branch.depth + 1))
        branches.append(Branch(splits=split2, id=next(ids), depth=branch.depth + 1))

    print("All branches pruned.")
    print("=" * 80)
    print(branch_counter)
    print("*" * 80)
    best_lb = min(pruned_lbs)
    return best_lb, best_ub, None, branch_lp_bounds

#FIXME watch out for new "branch_lp_bounds" return

if __name__== "__main__": 
    #arg parse
    pass
   