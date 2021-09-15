from sklearn import datasets  # 引入数据集
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt  # plt用于显示图片
from matplotlib import offsetbox


def calLDA(k):
    # LDA
    lda = LinearDiscriminantAnalysis(n_components=k).fit(data, label)  # n_components设置降维到n维度
    dataLDA = lda.transform(data)  # 将规则应用于训练集
    return dataLDA


def calPCA(k):
    # PCA
    pca = PCA(n_components=k).fit(data)
    # 返回测试集和训练集降维后的数据集
    dataPCA = pca.transform(data)
    return dataPCA


def draw():
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    fig = plt.figure('example', figsize=(11, 6))
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.xlim(xmax=9, xmin=-9)
    # plt.ylim(ymax=9, ymin=-9)
    color = ["red", "yellow", "blue", "green", "black", "purple", "pink", "brown", "gray", "Orange"]
    colors = []
    for target in label:
        colors.append(color[target])
    plt.subplot(121)
    plt.title("LDA 降维可视化")
    plt.scatter(dataLDA.T[0], dataLDA.T[1], s=10, c=colors)
    plt.subplot(122)
    plt.title("PCA 降维可视化")
    plt.scatter(dataPCA.T[0], dataPCA.T[1], s=10, c=colors)

    # plt.legend()
    plt.show()


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)  # 对每一个维度进行0-1归一化，注意此时X只有两个维度
    colors = ['#5dbe80', '#2d9ed8', '#a290c4', '#efab40', '#eb4e4f', '#929591', '#ababab', '#eeeeee', '#aaaaaa',
              '#213832']

    ax = plt.subplot()

    # 画出样本点
    for i in range(X.shape[0]):  # 每一行代表一个样本
        plt.text(X[i, 0], X[i, 1], str(label[i]),
                 # color=plt.cm.Set1(y[i] / 10.),
                 color=colors[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})  # 在样本点所在位置画出样本点的数字标签

    # 在样本点上画出缩略图，并保证缩略图够稀疏不至于相互覆盖
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # 假设最开始出现的缩略图在(1,1)位置上
        for i in range(data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)  # 算出样本点与所有展示过的图片（shown_images）的距离
            if np.min(dist) < 4e-3:  # 若最小的距离小于4e-3，即存在有两个样本点靠的很近的情况，则通过continue跳过展示该数字图片缩略图
                continue
            shown_images = np.r_[shown_images, [X[i]]]  # 展示缩略图的样本点通过纵向拼接加入到shown_images矩阵中

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(datasets.load_digits().images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)

    # plt.xticks([]), plt.yticks([])  # 不显示横纵坐标刻度
    if title is not None:
        plt.title(title)

    plt.show()


data = datasets.load_digits().data  # 一个数64维，1797个数
label = datasets.load_digits().target
dataLDA = calLDA(2)
dataPCA = calPCA(2)

# draw() #普通图


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plot_embedding(dataLDA, "LDA 降维可视化")
plot_embedding(dataPCA, "PCA 降维可视化")
