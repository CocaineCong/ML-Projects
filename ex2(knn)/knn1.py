"""
Sklearn中的make_circles方法生成训练样本
并随机生成测试样本，用KNN分类并可视化。
"""

from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import random

fig = plt.figure(1, figsize=(10, 5))
x1, y1 = make_circles(n_samples=400, factor=0.4, noise=0.1)
# 模型训练 求距离、取最小K个、求类别频率
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x1, y1)  # X是训练集(横纵坐标) y是标签类别

# 进行预测
x2 = random.random()  # 测试样本横坐标
y2 = random.random()  # 测试样本纵坐标

X_sample = np.array([[x2, y2]])  # 给测试点
# y_sample = knn.predict(X_sample)   # 调用knn进行predict得预测类别
y_sample = []
for i in range(0, 400):
    dx = x1[:, 0][i] - x2
    dy = x1[:, 1][i] - y2
    d = (dx ** 2 + dy ** 2) ** 1 / 2
    y_sample.append(d)

neighbors = knn.kneighbors(X_sample, return_distance=False)

plt.subplot(121)
plt.title('data by make_circles() 1')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', s=100, c=y1)
plt.scatter(x2, y2, marker='*', c='b')

plt.subplot(122)
plt.title('data by make_circles() 2')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', s=100, c=y1)
plt.scatter(x2, y2, marker='*', c='r', s=100)
for i in neighbors[0]:
    plt.scatter([x1[i][0], X_sample[0][0]], [x1[i][1], X_sample[0][1]], marker='o', c='b', s=100)

plt.show()
