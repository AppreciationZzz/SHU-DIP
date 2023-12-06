# 数字图像处理-实验6
# 任务1.1-用形态学方法去掉wire.tif 中的细线（可用opencv自带函数）
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ---------------------设计------------------------------------------
# 读取图像
image_cv = cv.imread('wire.tif', cv.IMREAD_GRAYSCALE)
image_plt = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

# 创建一个核（kernel）用于腐蚀和膨胀操作
kernel = np.ones((14, 14), np.uint8)
# 首先进行腐蚀操作，以去除细线
eroded = cv.erode(image_cv, kernel, iterations=1)
# 然后进行膨胀操作，以恢复原始图像的形状
dilated = cv.dilate(eroded, kernel, iterations=1)

# 保存处理后的图像
cv.imwrite('processed_wire.tif', dilated)
dilated_plt = cv.cvtColor(dilated, cv.COLOR_BGR2RGB)

# 生成最终效果图
plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(211)  # 创建一个包含多个子图的图形
plt.title('Original:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(image_plt)  # 原图

plt.subplot(212)
plt.title('Processed:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(dilated_plt)

plt.show()
