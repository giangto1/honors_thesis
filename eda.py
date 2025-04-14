import pyreadr
import numpy as np
from pgd import proximal_gradient_descent
import networkx as nx

result = pyreadr.read_r('Tau_Model.RData')
print("keys: ", result.keys())

r_df = result['suvr.f'] # shape: (364, 268)
y_df = result['YMat'] # shape: (364, 268)

r_i = r_df[0][0]
r_j = r_df[0][1]

y_i = y_df[0][0]
y_j = y_df[0][1]

D_ij = abs(y_i - y_j)
S_ij = abs(y_i + y_j)
X_ij = np.array([1, D_ij, S_ij])

B = np.random.rand(3)
print("B: ", B)
e_ij = 0

O_ij = np.dot(X_ij.T, B) + e_ij
print("O_ij: ", O_ij)
print("hello: ",np.sum(X_ij*B))
Graph = nx.path_graph(3)
# X: X_ij
# Y: O_ij
# B: B
# Graph: Graph
# alpha: 0.5
# lam: 0.1
B_opt, _, _ = proximal_gradient_descent(X_ij, O_ij, B, Graph, 0.5, 0.1)


print("X_ij: ", X_ij.shape)

print("B_opt: ", B_opt)

