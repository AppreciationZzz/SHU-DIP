# 数字图像处理-实验5
# 任务2.1-实现彩色图像的直方图均衡化
# 方法1：对RGB通道分别做直方图均衡化再合成
# 方法2：转换到HSV空间，仅对亮度分量V用直方图均衡化，再转换回RGB
# 注：OpenCv 读取图片是BGR的顺序，而下文首次编写是按RGB的通道顺次编写的，暂未测试。
# 注：下文凡是涉及OpenCv的颜色通道，顺序默认皆为 BGR 。涉及plt的颜色通道，顺序默认皆为 RGB 。
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# ---------------------设计彩色图像的直方图均衡化------------------------------------------
# 方法1：对RGB通道分别做直方图均衡化再合成
def histogram_equalization_color_image_through_RGB(image):
    blue_channel, green_channel, red_channel = cv.split(image)
    # 用OpenCV中对应函数实现直方图均衡化
    equ_blue_image = cv.equalizeHist(blue_channel)
    equ_green_image = cv.equalizeHist(green_channel)
    equ_red_image = cv.equalizeHist(red_channel)

    # 经过RGB均衡化的三个通道重新合并为一张图片,注：也许可用函数实现 np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape[:2]
    equalization_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将均衡化后的通道赋值给合并图像的相应通道
    equalization_image[:, :, 2] = equ_red_image  # 红色通道
    equalization_image[:, :, 1] = equ_green_image  # 绿色通道
    equalization_image[:, :, 0] = equ_blue_image  # 蓝色通道
    return equalization_image


# 方法2：转换到HSV空间，仅对亮度分量V用直方图均衡化，再转换回RGB
def histogram_equalization_color_image_through_HSV(image):
    # 将RGB图像转换为HSV颜色空间
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # value通道，即通过该通道提取亮度
    value_channel = hsv_image[:, :, 2]
    # 用OpenCV中对应函数实现直方图均衡化
    equ_value_image = cv.equalizeHist(value_channel)
    # 将均衡后value值赋于原通道
    hsv_image[:, :, 2] = equ_value_image
    # 将HSV颜色空间转回RGB图像，注：函数cvtColor会返回一个新的BGR格式的图像，而不会改变原始的 image 。
    equalization_image_through_hsv = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    return equalization_image_through_hsv


# ---------------------用图片进行测试------------------------------------------
# 导入原图
img_cv = cv.imread('mushroom.png')
img_plt = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

# 函数测试
# 法一：
equalization_img_through_RGB_cv = histogram_equalization_color_image_through_RGB(img_cv)
equalization_img_through_RGB_plt=cv.cvtColor(equalization_img_through_RGB_cv, cv.COLOR_BGR2RGB)
cv.imwrite('histogram_equalization_color_image_through_RGB.png', equalization_img_through_RGB_cv)
# 法二：
equalization_img_through_HSV_cv = histogram_equalization_color_image_through_HSV(img_cv)
equalization_img_through_HSV_plt=cv.cvtColor(equalization_img_through_HSV_cv, cv.COLOR_BGR2RGB)
cv.imwrite('histogram_equalization_color_image_through_HSV.png', equalization_img_through_HSV_cv)

# 生成最终效果图
plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(211)  # 创建一个包含多个子图的图形
# 第一个数字 2：表示整个图形分为2行。
# 第二个数字 1：表示整个图形分为1列。
# 第三个数字 1：表示当前子图位于第1个位置。
plt.title('Original:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_plt)  # 原图

plt.subplot(223)
plt.title('RGB:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(equalization_img_through_RGB_plt)

plt.subplot(224)
plt.title('HSV:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(equalization_img_through_HSV_plt)

plt.show()
