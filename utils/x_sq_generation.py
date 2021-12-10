from typing import Tuple, List
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from copy import deepcopy
__all__ = [
    'generate_X_SQ',
    'generate_graph',
    'get_V'
]


def get_combinations(d: int):
    from itertools import combinations
    return combinations(range(d), 2)


def get_outliers_score(data: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=30, random_state=10101)
    clf = clf.fit(data)
    scores = -clf.decision_function(data)
    return scores


def generate_X_SQ(data: np.ndarray, outliers_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
    n_features = data.shape[1]
    n_outliers = len(outliers_index)
    m = n_outliers
    n = (n_features * (n_features - 1)) >> 1
    X: np.ndarray = np.empty((m, n), dtype=np.float64)
    fp_record = get_combinations(n_features)
    retv_fp = deepcopy(fp_record)
    for idx, fp_id in enumerate(fp_record):
        scores = get_outliers_score(data[:, fp_id])[outliers_index]
        X[:, idx] = scores
    X_min = X.min()
    X_max = X.max()
    X: np.ndarray = (X - X_min) / (X_max - X_min)
    SQ = X.argsort(axis=1)[:, ::-1]
    return X, SQ, list(retv_fp)


def generate_graph(X: np.ndarray):
    m, n = X.shape
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for a, b in get_combinations(n):
        fp_a = X[:, a]
        fp_b = X[:, b]
        e_w = 1 / (1 + np.sqrt(np.square(fp_a - fp_b)).sum())
        g.add_edge(a, b, weight=e_w)
    return g


def get_V(G: nx.Graph) -> np.ndarray:
    node2vec = Node2Vec(G, seed=10101)
    word2vec = node2vec.fit()
    V = []
    for n in G.nodes:
        V.append(word2vec.wv[str(n)])
    return np.stack(V, axis=0)
