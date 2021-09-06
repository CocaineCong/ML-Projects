"""
Sklearn中的make_blobs方法生成数据，用K-Means聚类并可视化。
"""

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets


def create_data():
    X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1, 0], [5, 4], [2, 3], [10, 8], [7, 4]])
    return X, y

def init_centers(data, k):
    m, n = data.shape
    # m 样本个数，n特征个数
    center_ids = np.random.choice(m, k)
    centers = data[center_ids]
    return centers


def cal_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)


def kmeans_process(data, k):
    centers = init_centers(data, k)
    m, n = data.shape
    keep_changing = True
    pred_y = np.zeros((m,))

    iteration = 0
    while keep_changing:
        keep_changing = False
        # 计算剩余样本所属类别
        for i in range(m):
            min_distance = np.inf
            for center in range(k):
                distance = cal_dist(data[i, :], centers[center, :])
                if distance < min_distance:  # 判断离哪个更近
                    min_distance = distance
                    idx = center  # 类别换下
            if pred_y[i] != idx:  # 判断是否发生了改变
                keep_changing = True
            pred_y[i] = idx
        # 更新类别中心点坐标
        for center in range(k):
            cluster_data = data[pred_y == center]
            centers[center, :] = np.mean(cluster_data, axis=0)  # 求相同类别数据点的质心点
        print(centers)
        plt.clf()
        plt.title(f'iteration: {iteration}')
        plt.scatter(X[:, 0], X[:, 1], s=30, c=pred_y)
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c='k')
        plt.pause(1)
        iteration += 1

    return centers, pred_y


if __name__ == '__main__':
    X, y = create_data()
    plt.ion()
    centers, pred_y = kmeans_process(data=X, k=2)

    plt.ioff()
    plt.show()
