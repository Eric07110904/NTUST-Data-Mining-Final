# 2021 NTUST Data Mining Final Project

Implementation of *LP-Explain_Local_Pictorial_Explanation_for_Outliers* (https://ieeexplore.ieee.org/document/9338295) in Python


## Notice( important! )

這篇paper**沒有提供source code**，這個repository為我們用python實作的結果。

另外，因論文在outlier detector 的參數設定上沒有詳細說明，所以我們復現的成果會與論文圖不太一樣，但還是能成功找到合適的feature pair來視覺化outlier。


## Usage

This paper provide **six different datasets.**
1. airquality ([link](https://archive.ics.uci.edu/ml/datasets/Air+quality))
2. iris ([link](https://archive.ics.uci.edu/ml/datasets/Iris))
3. annthyroid ([link](http://odds.cs.stonybrook.edu/annthyroid-dataset/)) 
4. breastw ([link](http://odds.cs.stonybrook.edu/breast-cancer-wisconsin-original-dataset/))
5. mammograph ([link](http://odds.cs.stonybrook.edu/mammography-dataset/))
6. pendigits ([link](http://odds.cs.stonybrook.edu/pendigits-dataset/))

```shell=
python3 example.py -d <DATASET_NAME>
```

```shell=
python3 example.py -d pendigits
```

因為程式執行需花很長時間，所以**我們有提供已執行完畢的jupyter notebook (.ipynb)。**
方便在裡面看執行成果圖。

[Jupyter notebook link](./demo_notebooks)

## Experiment setting 


## Experiment result ( pendigits dataset、$\lambda = 3$)

### Heatmap plot 

This heatmap plot is used for feature pairs selection.

![lambda3 heatmap](./image/pendigit_3_heatmap.png)

### Scatter plot

![Cluster1](./image/pendigit_3_cluster1.png)

![Cluster1](./image/pendigit_3_cluster2.png)

![Cluster1](./image/pendigit_3_cluster3.png)

![Cluster1](./image/pendigit_3_cluster4.png)

![Cluster1](./image/pendigit_3_cluster5.png)

### Incrimination chart (Global、Local feature pairs)

![Cluster1](./image/pendigit_barchart.png)