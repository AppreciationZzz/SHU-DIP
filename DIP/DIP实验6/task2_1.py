# 数字图像处理-实验6
# 任务2.1-用一个20x20的全1结构元对 shape.tif 进行腐蚀、膨胀、开操作、闭操作（可用opencv自带函数）
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ---------------------设计------------------------------------------
# 读取图像
image_cv = cv.imread('shape.tif', cv.IMREAD_GRAYSCALE)
image_plt = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

# 创建一个20x20的全1结构元素
kernel = np.ones((20, 20), np.uint8)

# 腐蚀操作
eroded = cv.erode(image_cv, kernel, iterations=1)

# 膨胀操作
dilated = cv.dilate(image_cv, kernel, iterations=1)

# 开操作
opened = cv.morphologyEx(image_cv, cv.MORPH_OPEN, kernel)

# 闭操作
closed = cv.morphologyEx(image_cv, cv.MORPH_CLOSE, kernel)

eroded_plt = cv.cvtColor(eroded, cv.COLOR_BGR2RGB)
dilated_plt = cv.cvtColor(dilated, cv.COLOR_BGR2RGB)
opened_plt = cv.cvtColor(opened, cv.COLOR_BGR2RGB)
closed_plt = cv.cvtColor(closed, cv.COLOR_BGR2RGB)

# 生成最终效果图
plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(311)  # 创建一个包含多个子图的图形
plt.title('Original:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(image_plt)  # 原图

plt.subplot(323)
plt.title('eroded:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(eroded_plt)

plt.subplot(324)
plt.title('dilated:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(dilated_plt)

plt.subplot(325)
plt.title('opened:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(eroded_plt)

plt.subplot(326)
plt.title('closed:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(closed_plt)

plt.show()
