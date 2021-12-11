from copy import deepcopy
from typing import List, Tuple

import networkx as nx
import numpy as np
from node2vec import Node2Vec

__all__ = [
    'set_x_generation_paramters',
    'generate_X_SQ',
    'generate_graph',
    'get_V'
]

OUTLIERS_SCORE_KWARGS = {
    'n_estimators': 30,
    'random_state': 10101
}


def set_x_generation_paramters(*,
                               n_estimators=100,
                               max_samples="auto",
                               contamination="auto",
                               max_features=1.0,
                               bootstrap=False,
                               random_state=None):
    '''
    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    '''
    OUTLIERS_SCORE_KWARGS['n_estimators'] = n_estimators
    OUTLIERS_SCORE_KWARGS['max_samples'] = max_samples
    OUTLIERS_SCORE_KWARGS['contamination'] = contamination
    OUTLIERS_SCORE_KWARGS['max_features'] = max_features
    OUTLIERS_SCORE_KWARGS['bootstrap'] = bootstrap
    OUTLIERS_SCORE_KWARGS['random_state'] = random_state


def get_combinations(d: int):
    from itertools import combinations
    return combinations(range(d), 2)


def get_outliers_score(data: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(**OUTLIERS_SCORE_KWARGS)
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
