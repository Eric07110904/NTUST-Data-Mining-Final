from datetime import date
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from typing import Tuple
from numpy.typing import NDArray
import scipy.io
import pandas as pd 
import numpy as np 

# Air quality dataset 
def read_airquality(filename: str) -> Tuple[NDArray, NDArray, NDArray]:
    df = pd.read_csv(filename)
    date_list = df['Date'].values 
    time_list = df['Time'].values
    # data preprocess
    for index in range(len(date_list)):
        temp = date_list[index].split('/')
        temp2 = time_list[index].split(':')
        x = date(int(temp[2]),int(temp[1]),int(temp[0]))
        date_list[index] = x.toordinal()
        time_list[index] = temp2[0]
    data = df.values 
    # outlier detector 
    clf = IsolationForest(n_estimators=30, random_state=10101)
    pred = clf.fit_predict(data)
    inlier_index = np.where(pred[:] == 1)[0]
    outlier_index = np.where(pred[:] == -1)[0]
    return data, outlier_index, inlier_index

# Iris dataset 
def read_iris() -> Tuple[NDArray, NDArray, NDArray]:
    data = load_iris()['data']
    # outlier detector 
    clf = IsolationForest(max_samples=0.5, max_features=1.0, random_state=5)
    pred = clf.fit_predict(data) # -1 represent outlier
    inlier_index = np.where(pred[:] == 1)[0]
    outlier_index = np.where(pred[:] == -1)[0]
    return data, outlier_index, inlier_index

# Others dataset (breastw、pendigits、mammography、annthyroid)
def read_matfile(filename: str) -> Tuple[NDArray, NDArray, NDArray]:
    mat = scipy.io.loadmat(filename)
    data, label = mat['X'], mat['y']
    inlier_index = np.where(label[:, 0] == 0)[0]
    outlier_index = np.where(label[:, 0] == 1)[0]
    return data, outlier_index, inlier_index


