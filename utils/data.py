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
    'read_dataset'
]


Dataset = np.ndarray
InliersIndex = np.ndarray
OutliersIndex = np.ndarray


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
        date_list[index] = x.toordinal()
        time_list[index] = temp2[0]
    data = df.values
    # outlier detector
    clf = IsolationForest(n_estimators=30, random_state=10101)
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
