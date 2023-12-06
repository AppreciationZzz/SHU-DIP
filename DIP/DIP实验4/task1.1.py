# 数字图像处理-实验4
# 任务1.1-自己实现均值滤波
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# ---------------------设计均值滤波函数------------------------------------------
def Average_Filtering(img, size_kernel):
    # 创建一个输出数组以存储滤波后的图像
    img_average_filtering = np.zeros_like(img, dtype=np.uint8)

    # 计算核的中心距离
    pixel_distance = (size_kernel - 1) // 2  # //：这是整数除法运算符

    # 遍历图像像素
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 初始化用于计算平均值的变量
            pixel_sum = 0.0
            pixel_count = 0

            # 遍历核
            for k in range(-pixel_distance, pixel_distance + 1):
                for t in range(-pixel_distance, pixel_distance + 1):
                    # 检查当前核位置是否在图像边界内
                    if 0 <= (i + k) < img.shape[0] and 0 <= (j + t) < img.shape[1]:
                        pixel_sum += img[i + k][j + t]
                        pixel_count += 1

            # 计算平均值
            pixel_average = pixel_sum / pixel_count
            # 使用np.round四舍五入每个通道的像素值
            pixel_average = np.round(pixel_average).astype(np.uint8)
            # 将平均像素值赋给输出图像
            img_average_filtering[i][j] = pixel_average

    return img_average_filtering


# ---------------------用图片进行测试------------------------------------------
# 导入原图
img = cv.imread('Mona.jpg')

# 进行均值滤波
# img_average_filtering = cv.blur(img, (5, 5))
img_average_filtering = Average_Filtering(img, 5)
# 保存均值滤波后的图像
cv.imwrite('averaged_image.png', img_average_filtering)

# 生成子图
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.subplot(122), plt.imshow(img_average_filtering), plt.title('Average_filtering')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.show()
