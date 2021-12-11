import cvxpy as cp
import numpy as np

""" 
    1) fp_num: number of feature pairs 
    2) C: cluster from 'Self-Tuning Spectral Clustering'
    3) cluster_size: every cluster's size 
    4) cluster_num: number of clusters 
    5) W: weight matrix (Wj,i for the jth feature pair in the ith cluster)
    6) Z: Z's shape (cluster_num, cluster_siez[i], fp_num)
    7) summation: eq6's summation
    8) contraints: List of constraints 
    9) lbda: eq6's lambda
"""


def cvx_solver(C: list, X: np.ndarray, fp_num: int, lbda: int) -> np.ndarray:

    # parameters configuration:
    cluster_size = [len(c) for c in C]
    cluster_num = len(C)
    W = cp.Variable((fp_num, cluster_num))
    #Z = [cp.Variable((cluster_size[i], fp_num)) for i in range(cluster_num)]

    Z = [[] for i in range(cluster_num)]
    for i in range(cluster_num):
        for j in range(cluster_size[i]):
            t = []
            for k in range(fp_num):
                t.append(W[k][i])
            Z[i].append(t)
    
    theta = np.ones((cluster_num))
    summation = 0
    zeros = np.zeros((fp_num, cluster_num))
    ones = np.ones((fp_num, cluster_num))
    constraints = [W >= zeros, W <= ones, cp.mixed_norm(W, 2, 1) <= lbda]
    # sigma(sigma(theta(z * X.T)))
    for i in range(cluster_num):
        for ak in range(cluster_size[i]):
            summation += theta[i] * (Z[i][ak] @ X[C[i][ak]].T)
            # eq6 constraints
            temp_summation = 0
            for j in range(fp_num):
                temp_summation += Z[i][ak][j]
                constraints.append(Z[i][ak][j] <= W[j][i])
            constraints.append(temp_summation <= 1)
    # solve problem
    problem = cp.Problem(cp.Maximize(summation), constraints)
    problem.solve()
    return W.value
