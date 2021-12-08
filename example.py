from utils import *
from STSC.stsc import *

if __name__ == '__main__':
    iris_data, iris_outliers_index, iris_inliers_index = read_dataset('iris')
    X, SQ = generate_X_SQ(iris_data, iris_outliers_index)
    G = generate_graph(X)
    V = get_V(G)
    S = maximum_weight_matching(SQ, V)
    C = self_tuning_spectral_clustering_np(S)
    print(C)