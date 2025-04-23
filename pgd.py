import numpy as np
from NetworkLasso_solver import solve_NetworkLasso
# from NetworkLasso_solver_Hallac import solveX
import networkx as nx 
import cvxpy as cp
import time

from NetworkLasso_solver_Hallac import runADMM

from snap import *
from snap import TUNGraph
from snap import TIntFltH
from snap import TIntPr

import io

def nx_to_snap(Gnx):
    Gsnap = TUNGraph.New()
    for node in Gnx.nodes():
        Gsnap.AddNode(node)
    for u, v in Gnx.edges():
        Gsnap.AddEdge(u, v)
    return Gsnap

def g(X,Y,B):
    return 0.5 * np.sum(np.square(Y - np.sum(X*B,axis=1)))

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
    ps_time_per_iter = []
    while i < max_iters:
        print('i: ', i)
        t = 1
        obj_cur = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        print("obj: ", obj_cur)
        deriv_g = grad_g(X, Y, B_cur)
        start = time.time()
        proximal_operator = solve_NetworkLasso(B_cur - t * deriv_g, G=Graph, lam=2*t*lam, verbose=1) 
        duration = time.time()-start
        print("duration: ", duration)
        ps_time_per_iter.append(duration)
        G = (B_cur - proximal_operator[0]) / t
        
        keepgoing = True
        while keepgoing:
            if g(X, Y, B_cur - t * G) > g(X, Y, B_cur) - t * np.dot(deriv_g.flatten(), G.flatten()) + (t/2) * np.square(np.linalg.norm(G)):
                t = t * alpha
            else: 
                keepgoing = False
        B_cur = B_cur - t * G 

        i += 1

        obj_new = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        change = np.linalg.norm(obj_cur - obj_new)
        iter_values.append(obj_new)
        if change < 1e-4:
            break
    return B_cur, i, iter_values, ps_time_per_iter


def proximal_gradient_descent_hallac(X, Y, B, Graph, alpha, lam, max_iters=2000):
    t = 1
    i = 0
    B_cur = np.array(B)
    iter_values = []
    ps_time_per_iter = []
    while i < max_iters:
        print("i: ", i)
        t = 1
        obj_cur = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        print("obj: ", obj_cur)
        deriv_g = grad_g(X, Y, B_cur)
        Gsnap = nx_to_snap(Graph)
        n, p = B_cur.shape
        rho = 1
        numiters = 50
        useConvex = 1
        epsilon = 1e-3
        mu = 1e-3
        a = B_cur - t * deriv_g
        x0 = np.copy(a).T  # runADMM expects shape (p, n)
        u = np.zeros((p, 2 * Graph.number_of_edges()))
        z = np.zeros((p, 2 * Graph.number_of_edges()))
        edgeWeights = TIntPrFltH()
        for u_, v_ in Graph.edges():
            edgeWeights.AddDat(TIntPr(u_, v_), 1.0)
        start = time.time()
        x_admm, _, _, _ = runADMM(Gsnap, p, p, 2*t*lam, rho, numiters, x0, u, z, a.T, edgeWeights, useConvex, epsilon, mu)
        duration = time.time() - start
        ps_time_per_iter.append(duration)
        print("duration: ", duration)
        proximal_operator = (x_admm.T, None)

        G = (B_cur - proximal_operator[0]) / t
        
        keepgoing = True
        while keepgoing:
            if g(X, Y, B_cur - t * G) > g(X, Y, B_cur) - t * np.dot(deriv_g.flatten(), G.flatten()) + (t/2) * np.square(np.linalg.norm(G)):
                t = t * alpha
            else: 
                keepgoing = False
        B_cur = B_cur - t * G 

        i += 1

        obj_new = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        change = np.linalg.norm(obj_cur - obj_new)
        iter_values.append(obj_new)
        if change < 1e-4:
            break
    return B_cur, i, iter_values, ps_time_per_iter

import scs
import pandas as pd
def proximal_gradient_descent_cvxpy(X, Y, B, Graph, alpha, lam, max_iters=2000):
    i = 0
    B_cur = np.array(B)
    iter_values = []
    ps_time_per_iter = []
    while i < max_iters:
        print("i: ", i)
        t = 1
        obj_cur = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        print("obj: ", obj_cur)
        deriv_g = grad_g(X, Y, B_cur)
        start = time.time()
        proximal_operator = proximal_step_cvxpy(B_cur - t * deriv_g, Graph=Graph, lambda_=2*t*lam) 
        duration = time.time() - start
        print("duration: ", duration)
        ps_time_per_iter.append(duration)
        G = (B_cur - proximal_operator) / t
        keepgoing = True
        while keepgoing:
            if g(X, Y, B_cur - t * G) > g(X, Y, B_cur) - t * np.dot(deriv_g.flatten(), G.flatten()) + (t/2) * np.square(np.linalg.norm(G)):
                t = t * alpha
            else: 
                keepgoing = False
        B_cur = B_cur - t * G 
        # print('t: ', t)
        i += 1

        obj_new = g(X,Y,B_cur) + h(B_cur, lam, Graph)
        change = np.linalg.norm(obj_cur - obj_new)
        iter_values.append(obj_new)
        if change < 1e-4:
            break
    return B_cur, i, iter_values, ps_time_per_iter

def proximal_step_cvxpy(y, Graph, lambda_):
    n_nodes, p = y.shape
    x = cp.Variable((n_nodes, p))
    obj = cp.sum_squares(x - y)
    reg = 0
    for node1, node2 in Graph.edges():
        reg += cp.norm(x[node1] - x[node2],2)
    obj += lambda_ * reg

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.SCS,verbose=False)
    return x.value


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

# if __name__ == '__main__':
#     import copy
#     # Generate test data
#     print("For num nodes = 5, p = 3\n")
#     X, Y, B, Graph = generate_test_data(n_nodes=5, p=3)
#     print("Graph edges:", list(Graph.edges()))

#     origB = copy.deepcopy(B)
#     initB_pgd = copy.deepcopy(B)
#     initB_cvx = copy.deepcopy(B)
#     # Run proximal gradient descent
#     lambda_= 0.1

#     B_pgd, num_iters, iter_values = proximal_gradient_descent(X, Y, initB_pgd, Graph, alpha=0.5, lam=lambda_)
#     # Print results
#     print("Initial B:\n", origB)
#     print("\nOptimized B PGD:\n", B_pgd)
#     # print()
#     B_pgd_h, num_iters_h, iter_values_h = proximal_gradient_descent_hallac(X, Y, initB_pgd, Graph, alpha=0.5, lam=lambda_)
#     print("\nOptimized B PGD Hallac:\n", B_pgd_h)
#     # print()
#     B_cvx, obj_cvx, cvx_obj_values = proximal_gradient_descent_cvxpy(X, Y, B=initB_cvx, Graph=Graph, alpha=0.5, lam=lambda_)
#     # print("Initial B:\n", origB)
#     print("\nOptimized B CVX:\n", B_cvx)
#     print()

#     obj_pgd = g(X,Y,B_pgd) + h(B_pgd, lambda_, Graph)
#     obj_cvx = g(X,Y,B_cvx) + h(B_cvx, lambda_, Graph)
#     obj_pgd_h = g(X,Y,B_pgd_h) + h(B_pgd_h, lambda_, Graph)
#     # obj_cvx = 
#     print("Final objective PGD: ", obj_pgd)
#     print("Final objective CVX: ", obj_cvx)
#     print("Final objective PGD Hallac: ", obj_pgd_h)

#     # grad_norm = np.linalg.norm(grad_g(X, Y, B_pgd))
#     # print(f"\nGradient Norm at Final B: {grad_norm:.6f}")

#     # # progression plot
#     print("pgd progression: ", iter_values)
#     print("pgd hallac progression: ", iter_values_h)
#     # progression_dict = {'pgd': iter_values, "pgd_h": iter_values_h}
#     print("cvx progression: ", cvx_obj_values)

#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     plt.rcParams['text.usetex'] = False
#     plt.figure(figsize=(6, 4))
#     plt.plot(iter_values, label='pgd')
#     plt.plot(iter_values_h, label='pgd hallac')  # First line (auto-colored)
#     plt.plot(cvx_obj_values, label='cvx')  # Second line

#     # Add labels and legend
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('Objective progression')
#     plt.legend()
#     plt.grid(True)
#     # plt.tight_layout()

#     # Show the plot
#     plt.savefig('progression.png')

#     print("For num nodes = 50, p = 30\n")
#     X, Y, B, Graph = generate_test_data(n_nodes=50, p=30)
#     print("For num nodes = 500, p = 300\n")
#     X, Y, B, Graph = generate_test_data(n_nodes=500, p=300)

    
