# 数字图像处理-实验4
# 任务1.2-自己实现中值滤波
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# ---------------------设计中值滤波函数------------------------------------------
def Median_Filtering(img, size_kernel):
    # 创建一个输出数组以存储滤波后的图像
    img_median_filtering = np.zeros_like(img, dtype=np.uint8)

    # 计算核的中心距离
    pixel_distance = (size_kernel - 1) // 2

    # 遍历图像像素
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 初始化用于存储像素值的列表
            pixel_values = []

            # 遍历核
            for k in range(-pixel_distance, pixel_distance + 1):
                for t in range(-pixel_distance, pixel_distance + 1):
                    # 检查当前核位置是否在图像边界内
                    if 0 <= (i + k) < img.shape[0] and 0 <= (j + t) < img.shape[1]:
                        pixel_values.append(img[i + k][j + t])

            # 计算中值
            pixel_median = np.median(pixel_values)
            # 将中值赋给输出图像
            img_median_filtering[i][j] = pixel_median

    return img_median_filtering


# ---------------------用图片进行测试------------------------------------------
# 导入原图
img = cv.imread('Mona.jpg')

# 进行中值滤波
img_median_filtering = Median_Filtering(img, 5)
# 保存均值滤波后的图像
cv.imwrite('median_image.png', img_median_filtering)

# 生成子图
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.subplot(122), plt.imshow(img_median_filtering), plt.title('median_filtering')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.show()
