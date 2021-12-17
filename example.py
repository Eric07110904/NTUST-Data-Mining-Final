import os, argparse 
from utils import *
from STSC.stsc import *


if __name__ == '__main__':
    # set argparser 
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='please select one dataset!', required=True)
    args = parser.parse_args()
    dataset_name = args.dataset
    if dataset_name in ['breastw', 'annthyroid', 'pendigits', 'mammopraphy']:
        data, outliers_index, inliers_index = read_dataset('./dataset/'+dataset_name+'.mat')
    else:
        data, outliers_index, inliers_index = read_dataset(dataset_name)
    X, SQ, fp_record = generate_X_SQ(data, outliers_index)
    G = generate_graph(X)
    V = get_V(G)
    
    # save graph's weight 
    if not os.path.isfile('./weights/outlier_weight_{}.npy'.format(dataset_name)):
        S = maximum_weight_matching(SQ, V)
        np.save("./weights/outlier_weight_{}".format(dataset_name), S)
    else:
        S = np.load("./weights/outlier_weight_{}.npy".format(dataset_name))
    
    if dataset_name == 'airquality':
        C = self_tuning_spectral_clustering_np(S, max_n_cluster=13)   
    else:
        C = self_tuning_spectral_clustering_np(S) # for iris 
    print('number of clusters: ', len(C))

    # experiments on different lambda value (1、3、5、10)
    W_1 = cvx_solver(C, X, X.shape[1], 1)
    W_3 = cvx_solver(C, X, X.shape[1], 3)
    W_5 = cvx_solver(C, X, X.shape[1], 5)
    W_10 = cvx_solver(C, X, X.shape[1], 10)
    
    # draw different lambda's Heatmap 
    draw_heatmap(W_1.T) # lambda = 1  <====== global 
    draw_heatmap(W_3.T) # lambda = 3 
    draw_heatmap(W_5.T) # lambda = 5
    draw_heatmap(W_10.T)   # lambda = 10  <===== local
    
    # find the best feature pair for every cluster 
    cluster_fp_1 = get_clusters_fps(W_1.T, fp_record)
    cluster_fp_3 = get_clusters_fps(W_3.T, fp_record)
    cluster_fp_5 = get_clusters_fps(W_5.T, fp_record)
    cluster_fp_10 = get_clusters_fps(W_10.T, fp_record)
    
    df_list = [create_df(data[outliers_index], data[inliers_index], C[i]) for i in range(len(C))]
    W_1, _ = get_incrimination(C, X, X.shape[1] , W_1.T, 3)
    W_10, _ = get_incrimination(C, X, X.shape[1] , W_10.T, 3)
    save_W(W_10, './weights/W/', dataset_name)
    
    # draw different lambda's scatter
    draw_scatter(df_list, cluster_fp_1)
    draw_scatter(df_list, cluster_fp_3)
    draw_scatter(df_list, cluster_fp_5)
    draw_scatter(df_list, cluster_fp_5)
    
    # draw incrimination bar chart 
    draw_barchart(W_1, W_10)