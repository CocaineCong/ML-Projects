import numpy as np
from sklearn.decomposition import PCA
import os
import cv2 as cv
import matplotlib.pyplot as plt

h, w = 200, 180
path = 'face_images/'


def getinfo(path):
    TrainFiles = os.listdir(path)
    Train_Number = len(TrainFiles)
    train = []
    y_sample = []
    for k in range(0, Train_Number):
        Trainneed = os.listdir(path + '/' + TrainFiles[k])
        Trainneednumber = len(Trainneed)
        for i in range(0, Trainneednumber):
            image = cv.imread(path + '/' + TrainFiles[k] + '/' + Trainneed[i]).astype(np.float32)
            # image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            image = cv.cvtColor(image)
            train.append(image)
            y_sample.append(k)
    train = np.array(train)
    y_sample = np.array(y_sample)
    return train, y_sample


X, y = getinfo(path)
X_ = X.reshape(X.shape[0], h * w)
n_components = 10
pca = PCA(n_components).fit(X_)  # svd_solver='randomized', whiten=True
eigenfaces = pca.components_.reshape((n_components, h, w))

# 将输入数据投影到特征面正交基上
X_train_pca = pca.transform(X_)


def plot_gallery(images, titles, h, w, n_row=1, n_col=10):
    plt.figure(figsize=(1.8 * n_col, 2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)  # 图片位置布局
    for i in range(10):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap='gray')
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


eigenface_titles = ["%d" % (i + 1) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
