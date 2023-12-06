# 数字图像处理-实验5
# 任务1.1-将彩色图像的 RGB 3个通道 用灰度图显示出来
# 注：本文grey和gray混用，特此告知，前者为英式表达，后者为美式表达。
# 注：下文凡是涉及OpenCv的颜色通道，顺序默认皆为 BGR 。涉及plt的颜色通道，顺序默认皆为 RGB 。
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
from matplotlib import pyplot as plt


# ---------------------设计彩色图像转灰度图函数------------------------------------------
# 分割R通道，红色
def split_red(image):
    img_red_channel = image[:, :, 2]
    return img_red_channel


# 分割G通道，绿色
def split_green(image):
    img_green_channel = image[:, :, 1]
    return img_green_channel


# 分割B通道，蓝色
def split_blue(image):
    img_red_channel = image[:, :, 0]
    return img_red_channel


def RGB_red_to_grey(image):
    img_red_channel = split_red(image)
    cv.imwrite('grey_image_through_channel_red.png', img_red_channel)
    return img_red_channel


def RGB_green_to_grey(image):
    img_green_channel = split_green(image)
    cv.imwrite('grey_image_through_channel_green.png', img_green_channel)
    return img_green_channel


def RGB_blue_to_grey(image):
    img_blue_channel = split_blue(image)
    cv.imwrite('grey_image_through_channel_blue.png', img_blue_channel)
    return img_blue_channel


# ---------------------用图片进行测试------------------------------------------
# 导入原图
img_cv = cv.imread('araras.jpg')
img_plt =cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

# 按通道R进行灰度图转换
img_red_to_grey = RGB_red_to_grey(img_cv)
# 按通道G进行灰度图转换
img_green_to_grey = RGB_green_to_grey(img_cv)
# 按通道B进行灰度图转换
img_blue_to_grey = RGB_blue_to_grey(img_cv)

# 生成最终效果图
plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(211)  # 创建一个包含多个子图的图形
# 第一个数字 2：表示整个图形分为2行。
# 第二个数字 1：表示整个图形分为1列。
# 第三个数字 1：表示当前子图位于第1个位置。
plt.title('Original:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_plt)  # 原图

plt.subplot(234)
plt.title('channel_red_to grey')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_red_to_grey, cmap='gray')
#   在Matplotlib的imshow函数中，
#   即使你显示的是灰度图像，
#   它也可能会以伪彩色（pseudo-color）的方式呈现，这取决于颜色映射（colormap）的设置。

plt.subplot(235)
plt.title('channel_green_to grey')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_green_to_grey, cmap='gray')

plt.subplot(236)
plt.title('channel_blue_to grey')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_blue_to_grey, cmap='gray')

plt.show()
