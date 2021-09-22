from sklearn.datasets import load_digits
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ex1 import clustering_performance
import numpy as np
import os
import cv2 as cv


# KNN分类器
def test_KNN(*data):
    X_train, X_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_sample = knn.predict(X_test)
    print('KNN分类器')
    ACC = clustering_performance.cluster_acc(y_test, y_sample)
    print('Testing Score: %.4f' % ACC)
    return ACC


# 高斯贝叶斯分类器
def test_GaussianNB(*data):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()  # ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB','CategoricalNB']
    cls.fit(X_train, y_train)
    # print('高斯贝叶斯分类器')
    print('贝叶斯分类器')
    print('Testing Score: %.4f' % cls.score(X_test, y_test))
    return cls.score(X_test, y_test)


path_face = '../ex4_PCA/face_images/'
path_flower = '../ex5_LDA/flowers_new/'


# 读取Face image
def createDatabase(path):
    # 查看路径下所有文件
    TrainFiles = os.listdir(path)  # 遍历每个子文件夹
    # 计算有几个文件(图片命名都是以 序号.jpg方式)
    Train_Number = len(TrainFiles)  # 子文件夹个数
    X_train = []
    y_train = []
    # 把所有图片转为1维并存入X_train中
    for k in range(0, Train_Number):
        Trainneed = os.listdir(path + '/' + TrainFiles[k])  # 遍历每个子文件夹里的每张图片
        Trainneednumber = len(Trainneed)  # 每个子文件里的图片个数
        for i in range(0, Trainneednumber):
            image = cv.imread(path + '/' + TrainFiles[k] + '/' + Trainneed[i]).astype(np.float32)  # 数据类型转换
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # RGB变成灰度图
            X_train.append(image)
            y_train.append(k)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


X_train_flower, y_train_flower = createDatabase(path_flower)
X_train_flower = X_train_flower.reshape(X_train_flower.shape[0], 180*200)
X_train_flower, X_test_flower, y_train_flower, y_test_flower = \
    train_test_split(X_train_flower, y_train_flower, test_size=0.2, random_state=22)

digits = load_digits()
X_train_digits, X_test_digits, y_train_digits, y_test_digits = \
    train_test_split(digits.data, digits.target, test_size=0.2, random_state=22)

X_train_face, y_train_face = createDatabase(path_face)
X_train_face = X_train_face.reshape(X_train_face.shape[0], 180*200)
X_train_face, X_test_face, y_train_face, y_test_face = \
    train_test_split(X_train_face, y_train_face, test_size=0.2, random_state=22)

print('17flowers分类')
test_KNN(X_train_flower, X_test_flower, y_train_flower, y_test_flower)
test_GaussianNB(X_train_flower, X_test_flower, y_train_flower, y_test_flower)
print('Digits分类')
test_KNN(X_train_digits, X_test_digits, y_train_digits, y_test_digits)
test_GaussianNB(X_train_digits, X_test_digits, y_train_digits, y_test_digits)
print('Face images分类')
test_KNN(X_train_face, X_test_face, y_train_face, y_test_face)
test_GaussianNB(X_train_face, X_test_face, y_train_face, y_test_face)
