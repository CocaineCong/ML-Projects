from PIL import Image
import numpy as np
import os
from ex1.clustering_performance import clusteringMetrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def getImage(path):
    images = []
    for root, dirs, files in os.walk(path):
        if len(dirs) == 0:
            images.append([root + "\\" + x for x in files])
    return images


# 加载图片
images_files = getImage('face_images')
y = []
all_imgs = []
for i in range(len(images_files)):
    y.append(i)
    imgs = []
    for j in range(len(images_files[i])):
        img = np.array(Image.open(images_files[i][j]).convert("L"))  # 灰度
        # img = np.array(Image.open(images_files[i][j])) #RGB
        imgs.append(img)
    all_imgs.append(imgs)

# 可视化图片
w, h = 180, 200
pic_all = np.zeros((h * 10, w * 10))  # gray
for i in range(10):
    for j in range(10):
        pic_all[i * h:(i + 1) * h, j * w:(j + 1) * w] = all_imgs[i][j]
pic_all = np.uint8(pic_all)
pic_all = Image.fromarray(pic_all)
pic_all.show()

# 构造输入X
label = []
X = []
for i in range(len(all_imgs)):
    for j in all_imgs[i]:
        label.append(i)
        # temp = j.reshape(h * w, 3) #RGB
        temp = j.reshape(h * w)  # GRAY
        X.append(temp)


def keams_in(X_Data, k):
    kMeans1 = KMeans(k)
    y_p = kMeans1.fit_predict(X_Data)
    ACC, NMI, ARI = clusteringMetrics(label, y_p)
    t = "ACC:{},NMI:{:.4f},ARI:{:.4f}".format(ACC, NMI, ARI)
    print(t)
    return ACC, NMI, ARI


# PCA
def pca(X_Data, n_component, height, weight):
    X_Data = np.array(X_Data)
    pca1 = PCA(n_component)
    pca1.fit(X_Data)
    faces = pca1.components_
    faces = faces.reshape(n_component, height, weight)
    X_t = pca1.transform(X_Data)
    return faces, X_t


def draw(n_component, faces):
    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(hspace=0, wspace=0)
    for i in range(n_component):
        plt.subplot(2, 5, i + 1)
        plt.imshow(faces[i], cmap='gray')
        plt.title(i + 1)
        plt.xticks(())
        plt.yticks(())
    plt.show()


score = []
for i in range(10):
    _, X_trans = pca(X, i + 1, h, w)
    acc, nmi, ari = keams_in(X_trans, 10)
    score.append([acc, nmi, ari])

score = np.array(score)
bar_width = 0.25
x = np.arange(1, 11)
plt.bar(x, score[:, 0], bar_width, align="center", color="orange", label="ACC", alpha=0.5)
plt.bar(x + bar_width, score[:, 1], bar_width, color="blue", align="center", label="NMI", alpha=0.5)
plt.bar(x + bar_width*2, score[:, 2], bar_width, color="red", align="center", label="ARI", alpha=0.5)
plt.xlabel("n_component")
plt.ylabel("精度")
plt.legend()
plt.show()
