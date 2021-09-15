import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
from scipy.optimize import linear_sum_assignment as linear_assignment


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def splitImages(path):
    save_path = r'D:/CodeProjects/PycharmProjects/MLearningDeom/ML_Projects/ex5_LDA'
    TrainImages = os.listdir(path)
    Train_Number = len(TrainImages)
    os.mkdir(save_path + '/flowers_new')
    for i in range(1, 18):
        os.mkdir(save_path + '/flowers_new/' + str(range(1, 18)[i - 1]))
    for k in range(0, Train_Number):
        for i in range(1, 18):
            image = cv.imread(path + '/' + TrainImages[k])
            if k // 80 + 1 == range(1, 18)[i - 1]:
                cv.imwrite(save_path + '/flowers_new/' + str(k // 80 + 1) + '/' + TrainImages[k], image)


def translate(path):
    TrainFiles = os.listdir(path)
    Train_Number = len(TrainFiles)
    for k in range(0, Train_Number):
        Trainneed = os.listdir(path + '/' + TrainFiles[k])
        Trainneednumber = len(Trainneed)
        for i in range(0, Trainneednumber):
            image = cv.imread(path + '/' + TrainFiles[k] + '/' + Trainneed[i])
            p = cv.resize(image, (200, 180), interpolation=cv.INTER_CUBIC)
            cv.imwrite(path + '/' + TrainFiles[k] + '/' + Trainneed[i], p)


def createDatabase(path):
    TrainFiles = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    Train_Number = len(TrainFiles)
    x_train = []
    y_train = []
    for k in range(0, Train_Number):
        Trainneed = os.listdir(path + '/' + TrainFiles[k])
        Trainneednumber = len(Trainneed)
        for i in range(0, Trainneednumber):
            img = Image.open(path + '/' + TrainFiles[k] + '/' + Trainneed[i]).resize((image_size, image_size), Image.ANTIALIAS)
            to_image.paste(img, (i * image_size, k * image_size))
            image = cv.imread(path + '/' + TrainFiles[k] + '/' + Trainneed[i]).astype(np.float32)
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            x_train.append(image)
            y_train.append(k)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


if __name__ == '__main__':
    path = '17flowers'
    new_path = 'flowers_new'
    # splitImages(path)
    translate(new_path)
    image_size = 40
    image_column = 10
    image_row = 10
    to_image = Image.new('RGB', (image_column * image_size, image_row * image_size))
    x_train, y_train = createDatabase(new_path)
    x_train = x_train.reshape(x_train.shape[0], 180 * 200)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=22)

    plt.figure()
    plt.imshow(to_image)
    plt.show()

    A_PCA = []
    A_LDA = []
    for i in range(1, 10):
        # PCA + KNN
        pca = PCA(n_components=i).fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)
        knn = KNeighborsClassifier()
        knn.fit(x_train_pca, y_train)
        y_sample = knn.predict(x_test_pca)
        ACC_PCA = cluster_acc(y_test, y_sample)
        A_PCA.append(ACC_PCA)

        # LDA + KNN
        lda = LinearDiscriminantAnalysis(n_components=i).fit(x_train, y_train)
        x_train_lda = lda.transform(x_train)
        x_test_lda = lda.transform(x_test)
        knn = KNeighborsClassifier()
        knn.fit(x_train_lda, y_train)
        y_sample = knn.predict(x_test_lda)
        ACC_LDA = cluster_acc(y_test, y_sample)
        A_LDA.append(ACC_LDA)

    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.6
    index = np.arange(9)
    ax.set_xticks(index + bar_width / 2)

    cylinder1 = ax.bar(index, A_PCA, bar_width, alpha=opacity, color='r', label='PCA')
    cylinder2 = ax.bar(index + bar_width, A_LDA, bar_width, alpha=opacity, color='y', label='LDA')

    label = []
    for j in range(1, 10):
        label.append(j)
    ax.set_xticklabels(label)

    plt.ylabel('ACC')
    plt.xlabel('Component')
    ax.legend()
    plt.show()
