import numpy as np
import networkx as nx
from utils.x_sq_generation import get_combinations
import networkx.algorithms.matching as matching


def fp_simi(vec_fp_a, vec_fp_b):
    return 1 / (1 + np.sqrt(np.sum(np.power(vec_fp_a - vec_fp_b, 2))))

def maximum_weight_matching(SQ, V, l = 7, phi = 0.5):
    #V 為get_V輸出的結果, SQ為X_SQ_generation輸出的SQ, SQ要用argsort
    N = len(SQ)
    S: np.ndarray = np.ones((N, N) , dtype = np.float64)
    for a, b in get_combinations(N):
        BG = nx.Graph() # empty graph
        e_bp: np.ndarray = np.zeros((l, l) , dtype = np.float64)
        s_ab = 0
        for i, fp_a in enumerate(SQ[a]):
            for j, fp_b in enumerate(SQ[b]):
                e_bp[i][j] = fp_simi(V[fp_a], V[fp_b]) * np.power(phi, max(i, j))
                BG.add_edge(i, j + l, weight = e_bp[i][j]) #建立fp_a與fp_b之間的edge與權重
        M = matching.max_weight_matching(BG, maxcardinality = True)
        for i, j in M:
            i, j = min(i, j), max(i, j) - l
            s_ab += e_bp[i][j]
        s_ab *= (1 - phi)
        S[a][b] = S[b][a] = s_ab
    return S #Self-Tuning Spectral Clustering的輸入S