from scipy.cluster.vq import *
from pylab import *
from PIL import Image


def clusterpixels(infile, k, steps):
    im = array(Image.open(infile))
    dx = im.shape[0] / steps
    dy = im.shape[1] / steps
    features = []

    for x in range(steps):  # RGB三色通道
        for y in range(steps):
            R = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 0])
            G = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 1])
            B = mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 2])
            features.append([R, G, B])
    features = array(features, 'f')  # make into array
    # 聚类， k是聚类数目
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)
    codeim = code.reshape(steps, steps)
    codeim = np.array(Image.fromarray(codeim).resize((im.shape[1], im.shape[0])))
    return codeim


# k = 5
infile_Stones = 'stones.jpg'
im_Stones = array(Image.open(infile_Stones))
steps = (50, 100)  # image is divided in steps*steps region

# 显示原图
figure()
subplot(231)
title('original')
axis('off')
imshow(im_Stones)

for k in range(2, 7):
    codeim = clusterpixels(infile_Stones, k, steps[-1])
    subplot(2, 3, k)
    title('K=' + str(k))
    axis('off')
    imshow(codeim)

show()
