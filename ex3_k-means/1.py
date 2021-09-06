"""
Sklearn中的make_circles方法生成数据，用K-Means聚类并可视化。
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from ex1.clustering_performance import clusteringMetrics  # 导入的老师写的库

fig = plt.figure(1, figsize=(10, 5))
X1, y1 = make_circles(n_samples=400, factor=0.5, noise=0.1)
plt.subplot(121)
plt.title('original')
plt.scatter(X1[:, 0], X1[:, 1], c=y1)
plt.subplot(122)
plt.title('K-means')
kms = KMeans(n_clusters=2, max_iter=400)  # n_cluster聚类中心数 max_iter迭代次数
y1_sample = kms.fit_predict(X1, y1)  # 计算并预测样本类别
centroids = kms.cluster_centers_
plt.scatter(X1[:, 0], X1[:, 1], c=y1_sample)
plt.scatter(centroids[:, 0], centroids[:, 1], s=30, marker='*', c='b')

print(clusteringMetrics(y1, y1_sample))

plt.show()
