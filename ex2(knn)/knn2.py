"""
Sklearn中的datasets方法导入训练样本
并用留一法产生测试样本
用KNN分类并输出分类精度
"""
import warnings
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


iris = datasets.load_iris()
X = iris.data
y = iris.target

loo = LeaveOneOut()  # 留一法

K = []
Accuracy = []
for k in range(1, 16):
    correct = 0
    knn = KNeighborsClassifier(k)
    for train, test in loo.split(X):  # 对测试机和训练集进行分割
        knn.fit(X[train], y[train])
        y_sample = knn.predict(X[test])
        if y_sample == y[test]:
            correct += 1
    K.append(k)
    Accuracy.append(correct / len(X))
    plt.plot(K, Accuracy)
    plt.xlabel('Accuracy:')
    plt.ylabel('K:')
    print('K次数:{} Accuracy正确率:{}'.format(k, correct / len(X)))

plt.show()
