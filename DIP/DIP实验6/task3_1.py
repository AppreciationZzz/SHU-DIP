# 数字图像处理-实验6
# 任务3.1-用形态学方法，基于8连通对connect.png 中几个连通域进行标记（上色为不同颜色）（可用opencv自带函数）
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ---------------------设计------------------------------------------
# 读取图像,图像以灰度模式（黑白）加载
# 图像的颜色信息被抛弃，只保留灰度信息，像素值的范围被映射到 0 到 255 之间。
image_cv = cv.imread('connect.png', cv.IMREAD_GRAYSCALE)
image_plt = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

# 对图像进行反色处理，使背景变为黑色
inverted_image = cv.bitwise_not(image_cv)

# 进行八连通区域标记：connectivity=8
# num_labels:标记的总数,包括背景连通数-即计算结果要加1； 标签 0 通常表示背景，不需要上色
# labeled_image:标记后的图像
num_labels, labeled_image = cv.connectedComponents(inverted_image, connectivity=8)
# 在八连通区域标记中，背景通常被自动识别为像素值等于0（黑色）的区域，并且标记为0
# 所以根据原图背景是否为黑色进行反转，若背景为白色，进行反转；反之，若背景为黑色，不进行反转；

# 创建一个随机颜色映射
# 生成 num_labels 个随机颜色，每个颜色有三个通道（B、G、R）。
colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

# 创建一个白色图像（所有像素值为255），大小与原图像大小一致
colored_image = np.ones((image_cv.shape[0], image_cv.shape[1], 3), dtype=np.uint8) * 255  # 初始设置为白色背景

# 为每个不同的区域按照掩码mask上色
for label in range(1, num_labels):
    mask = labeled_image == label  # mask是一个二维布尔数组，
    # True 表示该像素属于标签为 label 的连通区域，而 False 表示不属于。
    colored_image[mask] = colors[label]

colored_image_plt=cv.cvtColor(colored_image, cv.COLOR_BGR2RGB)
# 生成最终效果图
plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(211)  # 创建一个包含多个子图的图形
plt.title('Original:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(image_plt)  # 原图

plt.subplot(212)
plt.title('Colored Image:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(colored_image_plt)

plt.show()
