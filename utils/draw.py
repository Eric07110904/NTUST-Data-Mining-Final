import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt 
import numpy as np 
from typing import List
__all__ = [
    'draw_heatmap',
    'draw_scatter',
    'get_clusters_fps',
    'bar_chart'
]

'''
    W_t: W transpose
    cluster_fp: every cluster's **best** fp
'''

def draw_heatmap(W_t: np.ndarray) -> None:
    plt.figure(figsize = (25,5))
    ax = sns.heatmap(W_t, linewidths=0.5, linecolor='white')

def draw_scatter(df_list: List, cluster_fp: List) -> None:
    for i in range(len(df_list)):
        plt.figure(figsize = (10,10))
        x_name = 'attr_' + str(cluster_fp[i][0][0])
        y_name = 'attr_' + str(cluster_fp[i][0][1])
        ax = sns.scatterplot(data=df_list[i], x=x_name, y=y_name, hue='label', style='label')
        ax.set_title("Cluster " + str(i+1)) 

def get_clusters_fps(W_t: np.ndarray, fp_record: List) -> List:
    cluster_fp = [[] for i in range(W_t.shape[0])]
    for i in range(W_t.shape[0]):
        for j in np.argsort(W_t[i])[-1:]: #取weight前n高的fp
            cluster_fp[i].insert(0, [fp_record[j][0], fp_record[j][1]])
    return cluster_fp

def bar_chart(lambda_1: np.ndarray, lambda_10: np.ndarray):
    size = len(lambda_10)
    y10 = np.array([1+i*3 for i in range(size)])
    y1 = np.array([2+i*3 for i in range(size)])
    plt.bar(y1 , lambda_1[:,1] , width=1, label="Global", color="orange")
    plt.bar(y10, lambda_10[:,1], width=1, label="Local", color="blue")
    plt.ylabel("Incrimination(%)") # y label
    plt.xlabel("Clusters") # x label
    plt.legend()

    temp = np.array([1.5+i*3 for i in range(size)])
    strTemp = np.array(["c"+str(i) for i in range(size)])
    plt.xticks(temp, strTemp)
    plt.show()

