import pyreadr
import numpy as np
from pgd import proximal_gradient_descent, proximal_gradient_descent_cvxpy
import networkx as nx

result = pyreadr.read_r('Tau_Model.RData')
print("keys: ", result.keys())

r_df = result['suvr.f'] # shape: (364, 268)
y_df = result['YMat'] # shape: (364, 268)
n,p = r_df.shape

print("r df: ",r_df.head())
print("y_df: ",y_df.head())

r_i = r_df[0] # region i for all k
r_j = r_df[1] # region j for all k

y_i = y_df[0]
y_j = y_df[1]

print("yi: ",y_i)
print("yj: ",y_j)

D_ij = abs(y_i.values - y_j.values)
print("D ij: ", D_ij.shape)

S_ij = abs(y_i.values + y_j.values)
print("S ij: ", S_ij.shape)

X = np.array([np.ones(n), D_ij, S_ij]).T
print("X shape: ", X.shape)
print("X ij: ", X)


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
print("rdf shape: ", r_df.shape)
Graph = nx.path_graph(n)
print("G: ", Graph)
# X: X_ij
# Y: O_ij
# B: B
# Graph: Graph
# alpha: 0.5
# lam: 0.1
B_yu, _, _ = proximal_gradient_descent(X, O, B, Graph, 0.5, 0.1)
B_cvx, _, _ = proximal_gradient_descent_cvxpy(X, O, B, Graph=Graph, alpha=0.5, lam=0.1)


print("X_ij: ", X.shape)

print("B_yu: ", B_yu)
print("B_cvx: ", B_cvx)

