# 数字图像处理-实验4
# 任务2.1-自己实现拉普拉斯算子
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# ---------------------设计拉普拉斯算子函数------------------------------------------
def Laplacian_Operator(img):
    # 创建一个输出数组以存储拉普拉斯算子处理后的图像
    img_laplacian = np.zeros_like(img, dtype=np.uint8)

    # 定义拉普拉斯算子核
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    # 获取图像的高度和宽度
    height, width = img.shape[:2]

    # 遍历图像像素
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 对每个像素应用拉普拉斯算子核
            laplacian_value = np.sum(img[i - 1:i + 2, j - 1:j + 2] * laplacian_kernel)
            # 将结果赋给输出图像
            # 使用 np.clip() 来确保其中的元素在0到255之间，
            # 小于0的值设置为0，大于255的值设置为255，而不会改变在0到255范围内的值。
            img_laplacian[i, j] = np.clip(laplacian_value, 0, 255).astype(np.uint8)

    return img_laplacian


# ---------------------用图片进行测试------------------------------------------
# 导入原图
img = cv.imread('blurry_moon.tif')

# 进行中值滤波
img_Laplacian_filtering = Laplacian_Operator(img)
# 保存均值滤波后的图像
cv.imwrite('Laplacian_image.png', img_Laplacian_filtering)

# 生成子图
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.subplot(122), plt.imshow(img_Laplacian_filtering), plt.title('Laplacian_filtering')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.show()
