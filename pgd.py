import numpy as np
from NetworkLasso_solver import solve_NetworkLasso
import networkx as nx 
import cvxpy as cp

def g(X,Y,B):
    return 0.5 * np.sum(np.square(Y - np.sum(X * B, axis=1)))

def h(B, lam, G):
    objtemp = 0
    for node1, node2 in G.edges():
        objtemp = objtemp + lam * np.linalg.norm(B[node1] - B[node2], ord=2)
    return objtemp
            

def grad_g(X,Y,B):
    grad = []
    for i in range(len(B)):
        grad.append((np.dot(X[i],B[i])-Y[i]) * X[i])
    return np.array(grad)

def proximal_gradient_descent(X, Y, B, Graph, alpha, lam, max_iters=2000):
    t = 1
    i = 0
    B_cur = np.array(B)
    iter_values = []
    while i < max_iters:
        obj_cur = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        print(f"Iteration {i}, obj_cur {obj_cur:.6f}")

        deriv_g = grad_g(X, Y, B_cur)
        proximal_operator = solve_NetworkLasso(B_cur - t * deriv_g, G=Graph, lam=t*lam) 
        G = (B_cur - proximal_operator[0]) / t
        
        keepgoing = True
        while keepgoing:
            if g(X, Y, B_cur - t * G) > g(X, Y, B_cur) - t * np.dot(deriv_g.flatten(), G.flatten()) + (t/2) * np.square(np.linalg.norm(G)):
                t = t * alpha
                # proximal_operator = solve_NetworkLasso(B_cur - t * deriv_g, G=Graph, lam=t*lam) 
                # G = (B_cur - proximal_operator[0]) / t
            else: 
                keepgoing = False
        # print("B cur: ", B_cur)
        print("t: ", t)
        # print("G: ", G)
        B_cur = B_cur - t * G 

        i += 1

        obj_new = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        change = obj_cur - obj_new
        iter_values.append(obj_new)
        if change < 1e-6:
            break
    return B_cur, i, iter_values




def solve_network_lasso_cvxpy(X, Y, B, Graph, lambda_):
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

    reg = 0
    for i, j in Graph.edges():
        reg += cp.norm(B[i] - B[j], 2)  
    reg *= lambda_
    # Define the full objective
    objective = cp.Minimize(loss + reg)

    # Solve the problem
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)  # SCS solver works well for L1 + L2 norms
    # print("Final Objective Value:", problem.value)
    # Return the optimized B matrix
    return B.value, problem.value


def generate_test_data(n_nodes=5, p=3):
    np.random.seed(42)  # For reproducibility
    
    # Feature matrix (n_nodes, p)
    X = np.random.randn(n_nodes, p)
    
    # Target values (n_nodes,)
    Y = np.random.randn(n_nodes)
    
    # Initial coefficient matrix B (n_nodes, p), randomly initialized
    B = np.random.randn(n_nodes, p)

    # Create a random network graph
    G = nx.path_graph(n_nodes)  # Simple chain graph


    return X, Y, B, G

if __name__ == '__main__':
    import copy
    # Generate test data
    X, Y, B, Graph = generate_test_data(n_nodes=5, p=3)
    print("Graph edges:", list(Graph.edges()))

    origB = copy.deepcopy(B)
    # Run proximal gradient descent
    lambda_= 0.1

    B_pgd, num_iters, iter_values = proximal_gradient_descent(X, Y, B.copy(), Graph, alpha=0.5, lam=lambda_)
    
    # Print results
    print("Initial B:\n", origB)
    print("\nOptimized B PGD:\n", B_pgd)
    # print("X shape:", X.shape)
    # print("Graph edges:", list(Graph.edges()))
    # # Y = Y.reshape(-1, 1)  # Ensure it's 2D
    # print("Y shape:", Y.shape)
    # print("B shape:", (X.shape[0], X.shape[1]))  # Expected shape of B


    B_cvx, obj_cvx = solve_network_lasso_cvxpy(X, Y, B=B.copy(), Graph=Graph, lambda_=lambda_)
    # print("Initial B:\n", origB)
    print("\nOptimized B CVX:\n", B_cvx)
    print()
    print("g(B_pgd): ", g(X,Y,B_pgd))
    print("g(B_cvx): ", g(X,Y,B_cvx))
    print()
    print("h(B_pgd): ", h(B_pgd, lambda_, Graph))
    print("h(B_cvx): ", h(B_cvx, lambda_, Graph))
    print()

    obj_pgd = g(X,Y,B_pgd) + h(B_pgd, lambda_, Graph)
    # obj_cvx = 
    print("Final objective PGD: ", obj_pgd)
    print("Final objective CVX: ", obj_cvx)

    grad_norm = np.linalg.norm(grad_g(X, Y, B_pgd))
    print(f"\nGradient Norm at Final B: {grad_norm:.6f}")

    import matplotlib.pyplot as plt

    differences = np.linalg.norm(B_pgd - B_cvx, axis=1)

    plt.figure(figsize=(6, 4))
    iter_lst = [i for i in range(num_iters)]
    print(len(iter_lst))
    print(len(iter_values))
    plt.plot(iter_values, label='iter vs obj val')
    plt.xlabel("Iteration Index")
    plt.ylabel("Objective Value PGD")
    plt.title(f"objective value minimization progression with lambda={lambda_}")
    plt.legend()
    # plt.plot(differences, 'ro-', label='Difference |B_PGD - B_CVX| per node')
    # plt.xlabel("Node Index")
    # plt.ylabel("Difference Magnitude")
    # plt.title("Difference Between PGD and CVXPY Solutions")
    # plt.legend()
    plt.show()

    

# Notes on what to say in meeting
# Tests: (n_node, p) = [(5,3),(10,7)]
# The code seems to work but larger graphs 