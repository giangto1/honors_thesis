import numpy as numpy
from pgd import proximal_gradient_descent
from pgd import proximal_gradient_descent_cvxpy
from pgd import proximal_gradient_descent_hallac
from pgd import generate_test_data
import time
import copy
import matplotlib.pyplot as plt

# n = 50, p = 30
X, Y, B, Graph = generate_test_data(n_nodes=50, p=50)

origB = copy.deepcopy(B)
initB_pgd = copy.deepcopy(B)
initB_cvx = copy.deepcopy(B)

lam = [0.01,0.1,1,10,100]
lam.reverse()
times_yu = []
times_hallac = []
times_cvx = []

for l in lam:
    start = time.time()
    B_pgd, num_iters, iter_values = proximal_gradient_descent(X, Y, initB_pgd, Graph, alpha=0.5, lam=l)
    elapsed_t = time.time() - start
    times_yu.append(elapsed_t)

    start = time.time()
    B_pgd_h, num_iters_h, iter_values_h = proximal_gradient_descent_hallac(X, Y, initB_pgd, Graph, alpha=0.5, lam=l)
    elapsed_t = time.time() - start
    times_hallac.append(elapsed_t)

    start = time.time()
    B_cvx, obj_cvx, cvx_obj_values = proximal_gradient_descent_cvxpy(X, Y, B=initB_cvx, Graph=Graph, alpha=0.5, lam=l)
    elapsed_t = time.time() - start
    times_cvx.append(elapsed_t)

print("yu: ", times_yu)
print("hallac: ", times_hallac)
print("cvx: ", times_cvx)

import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['text.usetex'] = False
plt.figure()
plt.semilogx(lam,times_yu, label='Yu, et al (2024)')
plt.semilogx(lam,times_hallac, label='Hallac, et al (2015)')  # First line (auto-colored)
plt.semilogx(lam,times_cvx, label='CVX custom solver')  # Second line

# Add labels and legend
plt.xlabel('lambda')
plt.ylabel('runtime')
plt.title('runtime efficiency comparison')
plt.legend()
plt.xticks(lam)
plt.grid(True)
# plt.show()

plt.savefig("test1.png")

