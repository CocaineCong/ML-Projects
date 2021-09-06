# -*- coding: utf-8 -*-
"""
Created on April 7, 2020

@author: Shiping Wang
  Email: shipingwangphd@gmail.com
  Date: April 14, 2020.
"""

from sklearn import metrics
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment  # 添加as语句不用修改代码中的函数名

'''
   Clustering accuracy
'''


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    sum = 0
    for i in range(len(ind[0])):
        j = ind[0][i]
        k = ind[1][i]
        sum += w[j, k]
    return sum * 1.0 / y_pred.size


'''
 Evaluation metrics of clustering performance
      ACC: clustering accuracy
      NMI: normalized mutual information
      ARI: adjusted rand index
'''


def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, predictiveLabel)

    # Normalized mutual information
    NMI = metrics.normalized_mutual_info_score(trueLabel, predictiveLabel)

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)

    return ACC, NMI, ARI
