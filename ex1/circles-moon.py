from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

fig = plt.figure(1)  # 创建一个figure的画图窗口
"""
n_samples：整数 生成的总点数，如果是奇数，内圆比外圆多一点
shuffle：布尔变量 是否打乱样本
noise：double或None 将高斯噪声的标准差加入到数据中
random_state：整数 RandomState instance or None，确定数据集变换和噪声的随机数生成。
factor：0 < double < 1 内外圆的半径之比
"""
x1, y1 = make_circles(n_samples=400, factor=0.2, noise=0.1)  # 生成一个二维的大圆，包含一个小圆
# datasets.make_circles()专门用来生成圆圈形状的二维样本
# factor表示内圈和外圈的半径之比.每圈共有n_samples/2个点
plt.subplot(1, 2, 1)  # 一行两列，这个画在第一个
plt.title('data by make_circles()')  # 标题
yList = []
for y in y1:
    if y == 1:
        y = 'r'
        yList.append(y)
    else:
        y = 'b'
        yList.append(y)
x_major_locator = MultipleLocator(0.5)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(-1.4, 1.4)         # 设置x轴间隔
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=yList)  # 横纵坐标

plt.tight_layout(pad=4)  # 两个图片之间距离

"""
n_numbers : 生成样本数量
shuffle : 是否打乱，类似于将数据集random一下
noise : 默认是false，数据集是否加入高斯噪声
random_state : 生成随机种子，给定一个int型数据，能够保证每次生成数据相同。
"""
plt.subplot(1, 2, 2)  # 一行两列，这个画在第二个
x2, y2 = make_moons(n_samples=400, noise=0.1)
x_major_locator2 = MultipleLocator(0.5)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator2)
plt.xlim(-1.4, 2.4)
plt.title('data by make_moons()')  # 标题
plt.scatter(x2[:, 0], x2[:, 1], marker='o', c=y2)
plt.show()
