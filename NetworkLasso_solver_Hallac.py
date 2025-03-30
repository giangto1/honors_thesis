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
    # g = sum_squares(xnew-a)+norm(xnew)
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
    # print(xnew.value, g.value)
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
