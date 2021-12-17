from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest

__all__ = [
    'read_airquality',
    'read_iris',
    'read_matfile',
    'read_dataset',
    'create_df',
    'save_W'
]


Dataset = np.ndarray
InliersIndex = np.ndarray
OutliersIndex = np.ndarray

def create_df(outliers: np.ndarray, inliers: np.ndarray, index: np.ndarray) -> pd.DataFrame:
    column_names = ['attr_' + str(i) for i in range(outliers.shape[1])]
    column_names.append('label')
    cluster_outliers = outliers[index]
    cluster_outliers = np.concatenate((cluster_outliers, np.ones((cluster_outliers.shape[0], 1))), axis=1)
    inliers = np.concatenate((inliers, np.zeros((inliers.shape[0], 1))), axis=1)
    df = pd.DataFrame(np.concatenate((inliers, cluster_outliers), axis=0), columns=column_names)
    return df  

def read_airquality() -> Tuple[Dataset, OutliersIndex, InliersIndex]:
    '''
    Air quality dataset
    '''
    df = pd.read_csv('dataset/AirQualityUCI_req.csv')
    date_list = df['Date'].values
    time_list = df['Time'].values
    # data preprocess
    for index in range(len(date_list)):
        temp = date_list[index].split('/')
        temp2 = time_list[index].split(':')
        x = date(int(temp[2]), int(temp[1]), int(temp[0]))
        date_list[index] = int(x.toordinal())
        time_list[index] = int(temp2[0])
    data = df.values
    # outlier detector
    clf = IsolationForest(n_estimators=20, random_state=10101)
    pred = clf.fit_predict(data)
    inlier_index = np.where(pred[:] == 1)[0]
    outlier_index = np.where(pred[:] == -1)[0]
    return data, outlier_index, inlier_index


def read_iris() -> Tuple[Dataset, OutliersIndex, InliersIndex]:
    '''
    Iris dataset
    '''
    data = load_iris()['data']
    # outlier detector
    clf = IsolationForest(max_samples=0.5, max_features=1.0, random_state=5)
    pred = clf.fit_predict(data)  # -1 represent outlier
    inlier_index = np.where(pred[:] == 1)[0]
    outlier_index = np.where(pred[:] == -1)[0]
    return data, outlier_index, inlier_index


def read_matfile(filename: str) -> Tuple[Dataset, OutliersIndex, InliersIndex]:
    '''
    Others dataset (breastw、pendigits、mammography、annthyroid)
    '''
    mat = loadmat(filename)
    data, label = mat['X'], mat['y']
    inlier_index = np.where(label[:, 0] == 0)[0]
    outlier_index = np.where(label[:, 0] == 1)[0]
    return data, outlier_index, inlier_index


def read_dataset(name: str) -> Tuple[Dataset, OutliersIndex, InliersIndex]:
    lower_name = name.lower()
    if lower_name == 'iris':
        return read_iris()
    elif lower_name == 'aq' or lower_name == 'airquality':
        return read_airquality()
    else:
        return read_matfile(name)


def save_W(data: np.ndarray, filepath: str, dataset_name: str) -> None:
    np.save(filepath + "weight_{}".format(dataset_name), data)