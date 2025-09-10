from micrograd.engine import Value
import cvxpy as cp



def planet_relaxation(output: Value, in_bounds, node_bounds):
    env = {}  # maps Value nodes to cp.Variable or float
    constraints = []

    # Traverse in topological order
    for v in output.compute_graph():
        if len(v.prev) == 0:
            # Input node
            if v in in_bounds:
                var = cp.Variable()
                env[v] = var
                constraints += [
                    var >= in_bounds[v].lower,
                    var <= in_bounds[v].upper,
                ]
            else:
                # Constant/weight node
                #print(f"Assigning constant: {v}, value type: {type(v.data)}") #debugging
                env[v] = v.data
                continue # do not add node bounds
        elif v.op == "+":
            z1, z2 = [env[p] for p in v.prev]
            var = cp.Variable()
            env[v] = var
            constraints.append(var == z1 + z2)
        elif v.op == "*":
            z1, z2 = [env[p] for p in v.prev]
            # print(f"Multiplying types: {type(a)}, {type(b)}") #debugging
            if isinstance(z1, (int, float)) and isinstance(z2, (int, float)):
                env[v] = z1 * z2
            else:
                var = cp.Variable()
                env[v] = var
                constraints.append(var == z1 * z2)
        elif v.op == "ReLU":
            assert len(v.prev) == 1, "ReLU needs to have exactly one input"
            input_node = v.prev[0]
            # Get input bounds for relaxation
            lb = node_bounds[input_node].lower
            ub = node_bounds[input_node].upper
            assert lb <= ub, "Lower bound must be less than or equal to upper bound"

            z = env[input_node]
            var = cp.Variable()
            env[v] = var
            # Standard PLANET ReLU relaxation
            if ub <= 0:
                # If upper bound is non-positive, ReLU is always zero
                constraints.append(var == 0)
            elif lb >= 0:
                # If lower bound is non-negative, ReLU is the input
                constraints.append(var == z)
            else:
                constraints += [
                    var >= 0,
                    var >= z,
                    var <= (ub / (ub - lb)) * (z - lb),
                ]
        else:
            raise NotImplementedError(f"Operation {v.op} not supported in PLANET relaxation.")
        
        if v in node_bounds:
            # add intermediate node bounds to aid the optimizer
            lb, ub = node_bounds[v]
            constraints += [
                env[v] >= lb,
                env[v] <= ub,
            ]

    prob_lower = cp.Problem(cp.Minimize(env[output]), constraints)
    try:
        result_lower = prob_lower.solve()
    except Exception as e:
        print(f"Solver failed: {e}")
        result_lower = None
    minimizer_lower = {in_node: env[in_node].value for in_node in in_bounds}
    
    return result_lower, minimizer_lower
