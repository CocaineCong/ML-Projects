import PIL.Image as Image
import os

IMAGES_PATH = r'D:\\data\\jpg\\'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
IMAGE_SIZE = 1000  # 每张小图片的大小
IMAGE_ROW = 5  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 10  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = r'D:\data\data.jpg'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
imageNames = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
              os.path.splitext(name)[1] == item]

# for x in os.listdir(IMAGES_PATH):
#     for item in IMAGES_FORMAT:

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(imageNames) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


# 定义图像拼接函数
def imageCompose():
    toImage = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + imageNames[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            toImage.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return toImage.save(IMAGE_SAVE_PATH)  # 保存新图


if __name__ == '__main__':
    imageCompose()  # 调用函数
