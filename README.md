# ML-Projects
记录机器学习的笔记


# 目录笔记
```
ML_Projects/
├── ex1
├── ex2(knn)
└── ex3(k-means)
```

- ex1: make_circle && make_moon 方法应用
- ex2: knn 算法实现
- ex3: k-means聚类算法

## ex1

左图是利用make_circle生成的随机散点图，右边是make_moon生成的

![在这里插入图片描述](https://img-blog.csdnimg.cn/996f17b53d464d52ac15c17b688454ae.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)

下载数据集，并进行数据集图片的拼接

![在这里插入图片描述](https://img-blog.csdnimg.cn/23ee0c0cfd33411ebc265a042186dae3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)

## ex2

使用knn算法对make_circle的散点图进行相邻k个点的预测

![在这里插入图片描述](https://img-blog.csdnimg.cn/836517fb09824a13845eed2060722580.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)

Sklearn中的datasets方法导入训练样本，并用留一法产生测试样本，用KNN分类并输出分类精度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/cf214ca0abbb402a9e446571c30e56f3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)

作图显示

![在这里插入图片描述](https://img-blog.csdnimg.cn/7ff44803b6ab4262b49f3d542d41044d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)

## ex3
###  K-Means 和 KNN 对比

| 名称   | K-Means   |  KNN   |
|----|----|----|
| 训练类型 | 无监督学习 | 监督学习|
| 计算样本范围 | 全局计算 | 邻居之间的局部计算|
| K的含义| K个聚类|K个近邻|
| 最终结果 | 只能知道是哪一类，不知道这个类的标签|明确知道了最后分类之后的标签类别|
|优点 | 原理比较简单，实现也是很容易，收敛速度快  |     简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归|
|缺点| 对K的取值、样本分布非常敏感|样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）|

### 缺陷
![在这里插入图片描述](https://img-blog.csdnimg.cn/2bafca42d22c459eaa238f1698bb1dcc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)


![在这里插入图片描述](https://img-blog.csdnimg.cn/1009f0ae15cf48988bd93f74ab2a7076.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5bCP55Sf5Yeh5LiA,size_20,color_FFFFFF,t_70,g_se,x_16)


###  K-Means 收敛过程

![请添加图片描述](https://img-blog.csdnimg.cn/b30d3ea64871431ea5587e69be29ce49.gif)



