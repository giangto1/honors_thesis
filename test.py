import numpy as np
import networkx as nx
import unittest
from NetworkLasso_solver import solve_NetworkLasso
import cvxpy as cp

# Importing the two solvers
from pgd import proximal_gradient_descent, solve_network_lasso_cvxpy, g, grad_g  

class TestNetworkLasso(unittest.TestCase):

    def setUp(self):
        """ Set up the same input for both solvers. """
        np.random.seed(42)  # Ensures reproducibility

        # Number of nodes and features
        self.n_nodes = 10
        self.p = 7
        self.lambda_ = 0.5  # Regularization parameter

        # Generate synthetic data
        self.X = np.random.randn(self.n_nodes, self.p)
        self.Y = np.random.randn(self.n_nodes)

        # Initialize B as zeros
        self.B = np.zeros((self.n_nodes, self.p))

        # Create a simple path graph (linear chain)
        self.G = nx.path_graph(self.n_nodes)

    def test_proximal_vs_cvxpy(self):
        """ Compare Proximal Gradient Descent against CVXPY solver. """
        
        # Solve using Proximal Gradient Descent
        B_pgd = proximal_gradient_descent(self.X, self.Y, self.B.copy(), self.G, alpha=0.1)

        # Solve using CVXPY
        B_cvxpy = solve_network_lasso_cvxpy(self.X, self.Y, self.G, self.lambda_)

        # Compute objective values
        obj_pgd = g(self.X, self.Y, B_pgd)
        obj_cvxpy = g(self.X, self.Y, B_cvxpy)

        # Compute Mean Squared Error (MSE) between solutions
        mse = np.mean((B_pgd - B_cvxpy) ** 2)

        print(f"Objective (PGD): {obj_pgd:.6f}")
        print(f"Objective (CVXPY): {obj_cvxpy:.6f}")
        print(f"Mean Squared Error between solutions: {mse:.6f}")

        # Assertions: Ensure objective values are close
        self.assertAlmostEqual(obj_pgd, obj_cvxpy, delta=1e-3, msg="Objective values should be close.")

        # Assertions: Ensure solutions are similar (low MSE)
        self.assertLess(mse, 1e-2, msg="MSE between solutions should be small.")

if __name__ == "__main__":
    unittest.main()
