import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import NGroupFL
import time
from libGroupFL import *


def solve_NetworkLasso(y, G, maxsteps=100, rho=1, lam=0.5, verbose=0):
    # n: number of all nodes, p: dimension of data point
    n, p = y.shape
    zeros = np.zeros((1, p))
    x = np.zeros(y.shape)

    # initial u,z, z1[k] return z_ij for (i,j) in E
    z1 = np.zeros((len(G.edges()), p))
    z2 = np.zeros((len(G.edges()), p))
    u1 = np.zeros((len(G.edges()), p))
    u2 = np.zeros((len(G.edges()), p))

    # list of all edges of G1
    Gedge = list(G.edges())
    Gedge.sort()

    # matrix A for Ax+Bz=C, for stopping criterion & dynamic rho
    # matrix A for Ax+Bz=C, for stopping criterion & dynamic rho
    if verbose:
        A=np.zeros((2*len(Gedge),n))
        for k in range(len(Gedge)):
            A[k,Gedge[k][0]]=1
            A[k+len(Gedge),Gedge[k][1]]=1
        r=1
        s=1

    dt = []
    obj = []
    objerr = []
    t = 0
    s = 1
    r = 1

    # for t in range(maxsteps):
    while (t <= maxsteps):
        # while (t<=maxsteps) and (s>1e-6 or r>1e-6):
        sys.stdout.write('\r' + 'network_lasso_explicit_status:' +
                         str(int(100 * t / maxsteps)) + '%')
        t += 1

        # varying rho if verbose = 1
        if verbose:
            if r > 10 * s:
                rho = 2 * rho
            elif s > 10 * r:
                rho = rho / 2

        t0 = time.time()

        # update for x
        for node in G.nodes():
            temp = np.zeros((1, p))
            for j in list(G.neighbors(node)):
                # print('node,j:',node,j)
                if j > node:
                    temp = temp + rho * \
                        z1[Gedge.index((node, j))] - rho * \
                        u1[Gedge.index((node, j))]
                else:
                    temp = temp + rho * \
                        z2[Gedge.index((j, node))] - rho * \
                        u2[Gedge.index((j, node))]

            x[node] = (2 * y[node] + temp) / \
                (2 + rho * len(list(G.neighbors(node))))
            # print('x:',x)

        # update for z1,z2
        tempz=np.concatenate((z1,z2))+0.0000
        # print('tempz:',tempz)

        for k in range(len(Gedge)):
            i = Gedge[k][0]
            j = Gedge[k][1]
            theta = max(
                1 - lam / rho / (np.linalg.norm(x[i] - x[j] + u1[k] - u2[k]) + 0.0000000001), 0.5)
            z1[k] = theta * (x[i] + u1[k]) + (1 - theta) * (x[j] + u2[k])
            z2[k] = (1 - theta) * (x[i] + u1[k]) + theta * (x[j] + u2[k])
            # print('theta:',theta)

        # update for u1,u2
        for k in range(len(Gedge)):
            i = Gedge[k][0]
            j = Gedge[k][1]
            u1[k] = u1[k] + (x[i] - z1[k])
            u2[k] = u2[k] + (x[j] - z2[k])

        # record runtime for each update
        t1 = time.time() - t0
        dt.append(t1)

        # stopping criterion
        if verbose:
            s_matrix = rho * np.dot(np.transpose(A), np.concatenate((z1,z2)) - tempz)
            r_matrix = np.dot(A, x) - np.concatenate((z1,z2))
            s = np.linalg.norm(s_matrix)
            r = np.linalg.norm(r_matrix)

        objtemp = 0.5 * np.sum(np.square(np.linalg.norm(x-y)))

        for node1, node2 in G.edges():
            objtemp = objtemp + lam * np.linalg.norm(x[node1] - x[node2])

        obj.append(objtemp)

    tevolution = []
    temp = 0

    for k in range(len(dt)):
        temp = temp + dt[k]
        tevolution.append(temp)

    for k in range(len(obj)):
        objerr.append(obj[k] - obj[-1])

    return x, obj, objerr, tevolution