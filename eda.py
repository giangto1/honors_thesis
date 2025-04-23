import pyreadr
import numpy as np
from pgd import proximal_gradient_descent, proximal_gradient_descent_cvxpy, proximal_gradient_descent_hallac
import networkx as nx
import time

result = pyreadr.read_r('Tau_Model.RData')
print("keys: ", result.keys())

# r_df = result['suvr.f'] # shape: (364, 268)
# y_df = result['YMat'] # shape: (364, 268)
# n,p = r_df.shape

# print("r df: ",r_df.head())
# print("y_df: ",y_df.head())

# r_i = r_df[0] # region i for all k
# r_j = r_df[1] # region j for all k

# y_i = y_df[0]
# y_j = y_df[1]

# print("yi: ",y_i)
# print("yj: ",y_j)

# D_ij = abs(y_i.values - y_j.values)
# print("D ij: ", D_ij.shape)

# S_ij = abs(y_i.values + y_j.values)
# print("S ij: ", S_ij.shape)

# X = np.array([np.ones(n), D_ij, S_ij]).T
# print("X shape: ", X.shape)
# print("X ij: ", X)
n = 364
X = np.random.randn(n,3)
B = np.random.randn(n,3)
print("B: ", B)

print("X dot B: ", X[0].dot(B[0]))
e_ij = 0.1 # np.random.normal()
print("e_ij: ", e_ij)
O = np.zeros(n)
for i in range(n):
    # for j in range(3):
    O_ij = X[i].dot(B[i]) + e_ij
    print("O_ij: ", O_ij)
    O[i] = O_ij

print("O_ij: ", O.shape)
print("hello: ",np.sum(X*B,axis=1))
# print("rdf shape: ", r_df.shape)
Graph = nx.path_graph(n)
print("G: ", Graph)
# X: X_ij
# Y: O_ij
# B: B
# Graph: Graph
# alpha: 0.5
# lam: 0.1
start_yu = time.time()
B_yu, _, iter_yu, times_yu = proximal_gradient_descent(X, O, B, Graph, 0.5,lam=0.1)
yu_duration = time.time() - start_yu
start_cvx = time.time()
B_cvx, _, iter_cvx, times_cvx = proximal_gradient_descent_cvxpy(X, O, B, Graph=Graph, alpha=0.5, lam = 0.1)
cvx_duration = time.time() - start_cvx
# start_hallac = time.time()
# B_pgd_h, _, iter_h, times_hallac = proximal_gradient_descent_hallac(X, O, B, Graph, alpha=0.5, lam=0.1)
# hallac_duration = time.time() - start_hallac
print("X_ij: ", X.shape)

print("B_yu: ", B_yu)
print("B_cvx: ", B_cvx)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(6, 4))
plt.plot(iter_yu, label='yu')
# plt.plot(iter_h, label='pgd hallac')  # First line (auto-colored)
plt.plot(iter_cvx, label='cvx')  # Second line

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Objective progression')
plt.legend()
plt.grid(True)
# plt.tight_layout()

# Show the plot
plt.savefig('eda_progression.png')


plt.figure(figsize=(6, 4))
plt.plot(times_yu, label='Time per iteration - Yu')
plt.plot(times_cvx, label='Time per iteration - CVX')
# plt.plot(times_hallac, label='Time per iteration - Hallac')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.title('Iteration Times')
plt.legend()
plt.grid(True)
plt.savefig('iteration_times.png')

plt.figure(figsize=(4, 4))
plt.bar(['Yu', 'CVX'], [yu_duration, cvx_duration])
plt.ylabel('Total Time (seconds)')
plt.title('Total Runtime Comparison')
plt.grid(axis='y')
plt.savefig('runtime_comparison.png')