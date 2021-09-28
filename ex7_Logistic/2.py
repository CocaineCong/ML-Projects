import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from ex1 import clustering_performance
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes

path = r'D:/腾讯/cifar-10-batches-py/'


def unpickle(file):  # 官方给的例程
    with open(file, 'rb') as fo:
        cifar = pickle.load(fo, encoding='bytes')
    return cifar


def test_LR(*data):
    X_train, X_test, y_train, y_test = data
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, y_train)
    # ACC = lr.score(X_test, y_test)
    print('逻辑回归分类器')
    print('Testing Score: %.4f' % lr.score(X_test, y_test))
    # print('Testing Score: %.4f' % ACC)
    return lr.score(X_test, y_test)


def test_GaussianNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()  # ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB','CategoricalNB']
    cls.fit(X_train, y_train)
    # print('高斯贝叶斯分类器')
    print('贝叶斯分类器')
    print('Testing Score: %.4f' % cls.score(X_test, y_test))
    return cls.score(X_test, y_test)


def test_KNN(*data):
    X_train, X_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_sample = knn.predict(X_test)
    print('KNN分类器')
    ACC = clustering_performance.cluster_acc(y_test, y_sample)
    print('Testing Score: %.4f' % ACC)
    return ACC


test_data = unpickle(path + 'test_batch')
for i in range(1, 4):
    train_data = unpickle(path + 'data_batch_' + str(i))
    X_train, y_train = train_data[b'data'][0:1234], np.array(train_data[b'labels'][0:1234])
    X_test, y_test = test_data[b'data'][0:1234], np.array(test_data[b'labels'][0:1234])
    print('data_batch_' + str(i))
    test_KNN(X_train, X_test, y_train, y_test)
    test_GaussianNB(X_train, X_test, y_train, y_test)
    test_LR(X_train, X_test, y_train, y_test)