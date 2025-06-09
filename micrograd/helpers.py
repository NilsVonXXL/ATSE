from micrograd.engine import Value
import cvxpy as cp


# Helper functions for micrograd branch and bound attempt



# Collect all ReLU nodes in the computation graph of the last score
def collect_relu_nodes(output_node, node_bounds, splitted_nodes=()):
    relu_nodes = []
    for v in output_node.compute_graph():
        if v.op == 'ReLU' and v not in splitted_nodes:
            # Only add if lower and upper bounds have a sign change
            v_inp = list(v.prev)[0]
            lower, upper = node_bounds[v_inp]
            if lower * upper < 0:
                relu_nodes.append(v)
    return relu_nodes


def planet_relaxation(output: Value, in_bounds, node_bounds):
    env = {}  # maps Value nodes to cp.Variable or float
    constraints = []

    # Traverse in topological order
    for v in output.compute_graph():
        if len(v.prev) == 0:
            # Input node
            if (v in in_bounds):
                #alternavit to lower != data 
                #(v.lower == -0.1 and v.upper == 0.1) or (v.lower == 0.4 and v.upper == 0.6)
                var = cp.Variable()
                env[v] = var
                constraints += [
                    var >= node_bounds[v].lower,
                    var <= node_bounds[v].upper,
                ]
            else:
                # Constant/weight node
                #print(f"Assigning constant: {v}, value type: {type(v.data)}") #debugging
                env[v] = v.data
        else:
            # Operation node
            if v.op == "+":
                a, b = [env[p] for p in v.prev]
                var = cp.Variable()
                constraints.append(var == a + b)
                env[v] = var
            elif v.op == "*":
                # For PLANET, only allow multiplication by constant (affine layers)
                a, b = [env[p] for p in v.prev]
                #print(f"Multiplying types: {type(a)}, {type(b)}") #debugging
                
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    raise NotImplementedError("PLANET relaxation does not support multiplication of two constants.")
                else:
                    var = cp.Variable()
                    constraints.append(var == a * b)
                    env[v] = var
                
            elif v.op == "ReLU":
                inp = [env[p] for p in v.prev][0]
                var = cp.Variable()
                # Get input bounds for relaxation
                input_node = list(v.prev)[0]
                l = node_bounds[input_node].lower
                u = node_bounds[input_node].upper
                # Standard PLANET ReLU relaxation
                if u <= 0:
                    # If upper bound is non-positive, ReLU is always zero
                    constraints.append(var == 0)
                elif l >= 0:
                    # If lower bound is non-negative, ReLU is the input
                    constraints.append(var == inp)
                else:
                    constraints += [
                        var >= 0,
                        var >= inp,
                        var <= (u / (u - l)) * (inp - l) if u > l else var <= 0,
                        var <= u if u > 0 else var <= 0,
                    ]
                env[v] = var
            else:
                raise NotImplementedError(f"Operation {v.op} not supported in PLANET relaxation.")

    prob_lower = cp.Problem(cp.Minimize(env[output]), constraints)
    result_lower = prob_lower.solve()
    
    prob_upper = cp.Problem(cp.Maximize(env[output]), constraints)
    result_upper = prob_upper.solve()
    
    return result_lower , result_upper
