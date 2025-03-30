import cvxpy as cp
import numpy as np
import networkx as nx

def solve_network_lasso_cvxpy(X, Y, Graph, lambda_):
    """
    Solves the Network Lasso problem using CVXPY.
    
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_nodes, p)
        Y (numpy.ndarray): Target values of shape (n_nodes,)
        Graph (networkx.Graph): Graph representing the network structure
        lambda_ (float): Regularization parameter

    Returns:
        B_opt (numpy.ndarray): Optimal B matrix of shape (n_nodes, p)
    """
    n_nodes, p = X.shape  # Number of nodes and features
    
    # Define the variable B (each row corresponds to a node)
    B = cp.Variable((n_nodes, p))

    # Quadratic loss term: 1/2 * sum ||Y_i - X_i^T B_i||^2
    loss = cp.sum_squares(Y - cp.sum(cp.multiply(X, B), axis=1)) / 2

    # Regularization term: sum ||B_i - B_j||_2 for (i, j) in the graph
    reg = 0
    for i, j in Graph.edges():
        reg += cp.norm(B[i] - B[j], 2)  # L2 norm

    # Define the full objective
    objective = cp.Minimize(loss + lambda_ * reg)

    # Solve the problem
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)  # SCS solver works well for L1 + L2 norms

    # Return the optimized B matrix
    return B.value


# Testing
n_nodes = 5
p = 3
X = np.random.randn(n_nodes, p)  # Random feature matrix
Y = np.random.randn(n_nodes)  # Random target values
Graph = nx.path_graph(n_nodes)  # Example: Chain structure graph
lambda_ = 0.5  # Regularization parameter
B_cvxpy = solve_network_lasso_cvxpy(X, Y, Graph, lambda_)
print("Optimized B from CVXPY:\n", B_cvxpy)

