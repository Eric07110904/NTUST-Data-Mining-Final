import os
import pickle
from utils import *
from STSC.stsc import *


if __name__ == '__main__':
    dataset_name = 'iris'
    data, outliers_index, inliers_index = read_dataset(dataset_name)
    X, SQ = generate_X_SQ(data, outliers_index)
    G = generate_graph(X)
    V = get_V(G)
    if not os.path.isfile('outlier_weight_{}.npy'.format(dataset_name)):
        S = maximum_weight_matching(SQ, V)
        np.save("outlier_weight_{}".format(dataset_name), S)
    else:
        S = np.load("outlier_weight_{}.npy".format(dataset_name))
    n_cluster = 8
    #C = self_tuning_spectral_clustering_np(S, n_cluster, n_cluster)
    C = self_tuning_spectral_clustering_np(S) # for iris 
    
    with open('{}_cluster_{}.pickle'.format(n_cluster, dataset_name), 'wb') as f:
        pickle.dump(C, f)
    print(len(C))
    
    W = cvx_solver(C, X, X.shape[1], 1)
    print(W.shape)
    print(W)