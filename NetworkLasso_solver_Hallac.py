import math
import time
import random
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from snap import *
from cvxpy import *
from numpy import linalg as LA
from NL_z_u_solvers import solveZ, solveU
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
# Other function in this folder
from multiprocessing import Pool

## EXTRA IMPORTS
import networkx as nx

def solveX(data):
    inputs = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    sizeData = int(data[data.size - 4])
    mu = data[data.size - 5]
    x = data[0:inputs]
    a = data[inputs:(inputs + sizeData)]
    neighs = data[(inputs + sizeData):data.size - 5]
    xnew = Variable(inputs)
    # print("input:",inputs)
    # print("a in pool:",a)

    # Fill in objective function here! Params: Xnew (unknown), a (side data at node)
    g = sum_squares(xnew - a)
    h = 0
    for i in range(int(neighs.size / (2 * inputs + 1))):
        weight = neighs[i * (2 * inputs + 1)]
        if(weight != 0):
            u = neighs[i * (2 * inputs + 1) + 1:i *
                       (2 * inputs + 1) + (inputs + 1)]
            z = neighs[i * (2 * inputs + 1) + (inputs + 1)
                            :(i + 1) * (2 * inputs + 1)]
            h = h + rho / 2 * square(norm(xnew - z + u))
    objective = Minimize(g + h)
    constraints = []
    p = Problem(objective, constraints)
    result = p.solve()
    if(result == None):
        # CVXOPT scaling issue. Rarely happens (but occasionally does when running thousands of tests)
        objective = Minimize(51 * g + 52 * h)
        p = Problem(objective, constraints)
        result = p.solve(verbose=False)
        if(result == None):
            print("SCALING BUG")
            objective = Minimize(52 * g + 50 * h)
            p = Problem(objective, constraints)
            result = p.solve(verbose=False)
    g_values = []
    # print("g value: ",g.value)
    
    #     g_values.append(i)
    # g = np.array(g_values)
    # total = np.sum(g) 
    # print('g: ', np.sum(g.value))
    return xnew.value, g.value


def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights, useConvex, epsilon, mu):
    # print("a:",a)
    nodes = G1.GetNodes()
    edges = G1.GetEdges()

    maxNonConvexIters = 6 * numiters

    # Find max degree of graph; hash the nodes
    (maxdeg, counter) = (0, 0)
    node2mat = TIntIntH()
    for NI in G1.Nodes():
        maxdeg = np.maximum(maxdeg, NI.GetDeg())
        node2mat.AddDat(NI.GetId(), counter)
        counter = counter + 1

    # Stopping criteria
    eabs = math.pow(10, -2)
    erel = math.pow(10, -3)
    (r, s, epri, edual, counter) = (1, 1, 0, 0, 0)
    A = np.zeros((2 * edges, nodes))
    for EI in G1.Edges():
        A[2 * counter, node2mat.GetDat(EI.GetSrcNId())] = 1
        A[2 * counter + 1, node2mat.GetDat(EI.GetDstNId())] = 1
        counter = counter + 1
    (sqn, sqp) = (math.sqrt(nodes * sizeOptVar), math.sqrt(2 * sizeOptVar * edges))

    # Non-convex case - keeping track of best point so far
    bestx = x
    bestu = u
    bestz = z
    bestObj = 0
    cvxObj = 10000000 * np.ones((1, nodes))
    if(useConvex != 1):
        # Calculate objective
        for i in range(G1.GetNodes()):
            bestObj = bestObj + cvxObj[0, i]
        for EI in G1.Edges():
            weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
            edgeDiff = LA.norm(x[:, node2mat.GetDat(
                EI.GetSrcNId())] - x[:, node2mat.GetDat(EI.GetDstNId())])
            bestObj = bestObj + lamb * weight * \
                math.log(1 + edgeDiff / epsilon)
        initObj = bestObj

    # Run ADMM
    iters = 0
    maxProcesses = 80
    pool = Pool(processes=np.minimum(np.maximum(nodes, edges), maxProcesses))
    # while(iters < numiters and (r > epri or s > edual or iters < 1)):
    dt = []
    obj = []
    tevolution = []
    while(iters < numiters):
        sys.stdout.write('\r'+'network_lasso_cvx_status:'+str(int(100*iters/numiters))+'%')
        # print("iters:",iters)

        # x-update
        neighs = np.zeros(((2 * sizeOptVar + 1) * maxdeg, nodes))
        edgenum = 0
        numSoFar = TIntIntH()
        t0 = time.time()
        for EI in G1.Edges():
            if (not numSoFar.IsKey(EI.GetSrcNId())):
                numSoFar.AddDat(EI.GetSrcNId(), 0)
            counter = node2mat.GetDat(EI.GetSrcNId())
            counter2 = numSoFar.GetDat(EI.GetSrcNId())
            neighs[counter2 * (2 * sizeOptVar + 1), counter] = edgeWeights.GetDat(
                TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
            neighs[counter2 * (2 * sizeOptVar + 1) + 1:counter2 * (2 *
                                                                   sizeOptVar + 1) + (sizeOptVar + 1), counter] = u[:, 2 * edgenum]
            neighs[counter2 * (2 * sizeOptVar + 1) + (sizeOptVar + 1):(counter2 + 1)
                   * (2 * sizeOptVar + 1), counter] = z[:, 2 * edgenum]
            numSoFar.AddDat(EI.GetSrcNId(), counter2 + 1)

            if (not numSoFar.IsKey(EI.GetDstNId())):
                numSoFar.AddDat(EI.GetDstNId(), 0)
            counter = node2mat.GetDat(EI.GetDstNId())
            counter2 = numSoFar.GetDat(EI.GetDstNId())
            neighs[counter2 * (2 * sizeOptVar + 1), counter] = edgeWeights.GetDat(
                TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
            neighs[counter2 * (2 * sizeOptVar + 1) + 1:counter2 * (
                2 * sizeOptVar + 1) + (sizeOptVar + 1), counter] = u[:, 2 * edgenum + 1]
            neighs[counter2 * (2 * sizeOptVar + 1) + (sizeOptVar + 1):(counter2 + 1)
                   * (2 * sizeOptVar + 1), counter] = z[:, 2 * edgenum + 1]
            numSoFar.AddDat(EI.GetDstNId(), counter2 + 1)

            edgenum = edgenum + 1

        temp = np.concatenate((x, a, neighs, np.tile(
            [mu, sizeData, rho, lamb, sizeOptVar], (nodes, 1)).transpose()), axis=0)
        values = pool.map(solveX, temp.transpose())
        newx = np.array(values)[:, 0].tolist()
        newcvxObj = np.array(values)[:, 1].tolist()
        # print("newcvxObj:", newcvxObj)
        # x = np.array(newx).transpose()[0]
        # print("newx:",newx)
        x = np.array(newx).transpose()
        # print("Size of x:",x.shape,"x:",x)

        # cvxObj = np.reshape(np.array(newcvxObj), (-1, nodes))
        # print("cvxObj:",cvxObj)
        # z-update
        ztemp = z.reshape(2 * sizeOptVar, edges, order='F')
        utemp = u.reshape(2 * sizeOptVar, edges, order='F')
        xtemp = np.zeros((sizeOptVar, 2 * edges))
        counter = 0
        weightsList = np.zeros((1, edges))
        for EI in G1.Edges():
            xtemp[:, 2 *
                  counter] = np.array(x[:, node2mat.GetDat(EI.GetSrcNId())])
            xtemp[:, 2 * counter + 1] = x[:, node2mat.GetDat(EI.GetDstNId())]
            weightsList[0, counter] = edgeWeights.GetDat(
                TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
            counter = counter + 1
        xtemp = xtemp.reshape(2 * sizeOptVar, edges, order='F')
        temp = np.concatenate((xtemp, utemp, ztemp, np.reshape(weightsList, (-1, edges)), np.tile(
            [epsilon, useConvex, rho, lamb, sizeOptVar], (edges, 1)).transpose()), axis=0)
        newz = pool.map(solveZ, temp.transpose())
        ztemp = np.array(newz).transpose()[0]
        ztemp = ztemp.reshape(sizeOptVar, 2 * edges, order='F')
        # For dual residual
        s = LA.norm(rho * np.dot(A.transpose(), (ztemp - z).transpose()))
        z = ztemp

        # u-update
        (xtemp, counter) = (np.zeros((sizeOptVar, 2 * edges)), 0)
        for EI in G1.Edges():
            xtemp[:, 2 *
                  counter] = np.array(x[:, node2mat.GetDat(EI.GetSrcNId())])
            xtemp[:, 2 * counter + 1] = x[:, node2mat.GetDat(EI.GetDstNId())]
            counter = counter + 1
        temp = np.concatenate(
            (u, xtemp, z, np.tile(rho, (1, 2 * edges))), axis=0)
        newu = pool.map(solveU, temp.transpose())
        u = np.array(newu).transpose()

        # Update best objective (for non-convex)
        if(useConvex != 1):
            tempObj = 0
            # Calculate objective
            for i in range(G1.GetNodes()):
                tempObj = tempObj + cvxObj[0, i]
            initTemp = tempObj
            for EI in G1.Edges():
                weight = edgeWeights.GetDat(
                    TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
                edgeDiff = LA.norm(x[:, node2mat.GetDat(
                    EI.GetSrcNId())] - x[:, node2mat.GetDat(EI.GetDstNId())])
                tempObj = tempObj + lamb * weight * \
                    math.log(1 + edgeDiff / epsilon)
            # Update best variables
            if(tempObj <= bestObj):
                bestx = x
                bestu = u
                bestz = z
                bestObj = tempObj
                print("Iteration ", iters, "; Obj = ",
                      tempObj, "; Initial = ", initTemp)

            if(iters == numiters - 1 and numiters < maxNonConvexIters):
                if(bestObj == initObj):
                    numiters = numiters + 1

        # Stopping criterion - p19 of ADMM paper
        epri = sqp * eabs + erel * \
            np.maximum(LA.norm(np.dot(A, x.transpose()), 'fro'),
                       LA.norm(z, 'fro'))
        edual = sqn * eabs + erel * \
            LA.norm(np.dot(A.transpose(), u.transpose()), 'fro')
        r = LA.norm(np.dot(A, x.transpose()) - z.transpose(), 'fro')
        s = s

        #print r, epri, s, edual
        t1 = time.time() - t0
        dt.append(t1)

        # objtemp = (LA.norm(x-a))**2+LA.norm(x)
        objtemp = (LA.norm(x - a))**2
        for edge in G1.Edges():
            node1 = edge.GetSrcNId()
            node2 = edge.GetDstNId()
            objtemp = objtemp + lamb * LA.norm(x[:, node1] - x[:, node2])

        obj.append(objtemp)

        iters = iters + 1

    pool.close()
    pool.join()

    objerror = []
    temp = 0

    for k in range(numiters):
        temp = temp + dt[k]
        tevolution.append(temp)

    for k in range(numiters):
        objerror.append(np.absolute(obj[k] - obj[-1]))

    return x, tevolution, obj, objerror


## ADDED CODE TO ORIGINAL CODE ##
def g(X,Y,B):
    return 0.5 * np.sum(np.square(Y - np.sum(X*B, axis=1)))

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

def nx_to_snap(Gnx):
    Gsnap = TUNGraph.New()
    for node in Gnx.nodes():
        Gsnap.AddNode(node)
    for u, v in Gnx.edges():
        Gsnap.AddEdge(u, v)
    return Gsnap

if __name__ == '__main__':
    n_nodes=5
    p=3
    np.random.seed(42)  # For reproducibility
    
    X = np.array([[ 0.49671415, -0.1382643,   0.64768854],[ 1.52302986, -0.23415337, -0.23413696],[ 1.57921282,  0.76743473, -0.46947439],[ 0.54256004, -0.46341769, -0.46572975],[ 0.24196227, -1.91328024, -1.72491783]])
    
    Y =np.array([-0.56228753, -1.01283112,  0.31424733, -0.90802408, -1.4123037 ])
    
    B = np.array([[ 1.46564877, -0.2257763,   0.0675282 ],[-1.42474819, -0.54438272,  0.11092259],[-1.15099358,  0.37569802, -0.60063869],[-0.29169375, -0.60170661,  1.85227818],[-0.01349722, -1.05771093,  0.82254491]])

    G = nx.path_graph(n_nodes)  # Simple chain graph
    lambda_= 0.1
    Gsnap = nx_to_snap(G)
    n, p = B.shape
    rho = 1
    numiters = 50
    useConvex = 1
    epsilon = 1e-3
    mu = 1e-3
    t = 1
    deriv_g = grad_g(X, Y, B)
    a = B - t * deriv_g
    x0 = np.copy(a).T  # runADMM expects shape (p, n)
    u = np.zeros((p, 2 * G.number_of_edges()))
    z = np.zeros((p, 2 * G.number_of_edges()))
    edgeWeights = TIntPrFltH()
    for u_, v_ in G.edges():
        edgeWeights.AddDat(TIntPr(u_, v_), 1.0)

    x, tevolution, obj, objerror = runADMM(Gsnap, p, p, 2*t*lambda_, rho, numiters, x0, u, z, a.T, edgeWeights, useConvex, epsilon, mu)
    proximal_operator = (x.T, None)
    print(x)
    # print(tevolution)
    print(len(obj))
    print(obj)
    print("final obj: ", obj[-1])