# 数字图像处理-实验7
# 任务1.1-自己编程实现书本10.3.2（基本的全局阈值处理）和10.3.3（最优全局阈值处理）中提到的两种分割方法，
# 对rice.tif，finger.tif和poly.tif进行分割，并对比结果。
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# ---------------------设计切割函数------------------------------------------
# 基本的全局阈值处理
def Basic_Global_Thresholding(img):
    # 读取图像
    image = img

    # 检查图像是否成功加载
    if image is None:
        print("无法加载图像")

    else:
        # 定义迭代算法的阈值收敛条件
        epsilon = 1e-100  # 差值小于epsilon时停止迭代

        # 初始化阈值
        T = 169  # 可以选择任何合适的初始值

        # 迭代体
        while True:
            # 使用当前阈值T分割图像
            G1 = image[image > T]
            G2 = image[image <= T]

            # 检查是否有像素值符合条件，避免阈值划分极端，从而提前终止迭代
            if G1.size == 0 or G2.size == 0:
                break

            # 计算G1和G2的平均灰度值
            m1 = np.mean(G1)
            m2 = np.mean(G2)

            # 计算新的阈值
            new_T = (m1 + m2) / 2

            # 检查是否达到收敛条件；abs：返回一个数的绝对值
            if abs(T - new_T) < epsilon:
                break

            T = new_T  # 更新阈值

        # 将图像二值化
        binary_image = (image > T).astype(np.uint8) * 255
        # (image > T)：这部分代码创建一个布尔数组
        # .astype(np.uint8)：将布尔数组类型转换为 uint8 数据类型，即无符号8位整数类型。True转换为1，False转换为0。
        return binary_image


# 最优全局阈值处理
def Otsu_Thresholding(img):
    image = img

    # 计算图像的归一化直方图
    hist, bins = np.histogram(image, bins=256, range=(0, 256), density=True)
    L = len(hist)

    # 计算累积和p1(k)和累积均值m(k)
    p1 = np.zeros(L)
    m = np.zeros(L)
    # 初始化起始值
    p1[0] = hist[0]  # hist[0] 表示直方图中的第一个分量，通常对应于灰度级为0。
    m[0] = 0  # 计算累积均值时，初始的累积均值为0
    # 循环迭代求累计和p1和累计均值m
    for k in range(1, L):
        p1[k] = p1[k - 1] + hist[k]
        m[k] = m[k - 1] + k * hist[k]

    # 计算全局灰度均值mG，数组m的最后一个元素 m[L-1] 表示了所有灰度级的累积均值，因此它是整个图像的灰度平均值。
    mG = m[L - 1]

    # 计算类间方差
    sigma_b = np.zeros(L)
    k_stars = []  # 保存单个/多个 k* 值
    max_sigma_b = 0

    for k in range(0, L):
        if 0 < p1[k] < 1:  # 检查是否满足条件以计算类间方差。
            # 条件是 p1[k]（背景像素的累积概率）必须大于0且小于1。
            # 确保前景和背景都包含像素。
            p2 = 1 - p1[k]  # 前景像素的累积概率
            # average_gray_p1 ：背景像素的平均灰度值；average_gray_p2 ：前景像素的平均灰度值
            # 课本提供的类间方差的计算公式：
            sigma_b[k] = ((mG * p1[k] - m[k]) ** 2) / (p1[k] * (1 - p1[k]))
            # 非书本计算方式：
            # 前景/背景像素和所有像素的权重分布不同
            # average_gray_p1 = m[k] / p1[k]
            # average_gray_p2 = (mG - m[k]) / p2
            # sigma_b[k] = p1[k] * p2 * (average_gray_p1 - average_gray_p2) ** 2

            if sigma_b[k] > max_sigma_b:
                max_sigma_b = sigma_b[k]
                k_stars = [k]  # 保存当前的 k* 值
            # 类间方差的极大值不唯一时，取各个极大值的k值的平均，赋给k*
            elif sigma_b[k] == max_sigma_b:
                k_stars.append(k)  # 如果有相等的类间方差，添加到 k_stars 中
    # 计算 k* 平均值
    k_star_avg = np.mean(k_stars)

    # 计算全局方差
    sigma_global = 0
    for i in range(0, L):
        sigma_global = sigma_global + ((i - mG) ** 2) * hist[i]

    # 计算可分离性测度;separability_measure:可分离性测度
    separability_measure = max_sigma_b / sigma_global

    # 使用最佳阈值分割图像
    binary_image = (image > k_star_avg).astype(np.uint8) * 255
    return binary_image, separability_measure


# ---------------------调用函数实现切割------------------------------------------
img_cv = cv.imread('poly.tif', cv.IMREAD_GRAYSCALE)
image_plt = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)

img_Basic = Basic_Global_Thresholding(img_cv)
img_Basic_plt = cv.cvtColor(img_Basic, cv.COLOR_BGR2RGB)
img_Otsu, separability_measure = Otsu_Thresholding(img_cv)
img_Otsu_plt = cv.cvtColor(img_Otsu, cv.COLOR_BGR2RGB)
print("可分离性测度:", separability_measure)

# 生成最终效果图
plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(211)  # 创建一个包含多个子图的图形
plt.title('Original:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(image_plt)  # 原图

plt.subplot(223)
plt.title('Basic Global Thresholding:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_Basic_plt)

plt.subplot(224)
plt.title('Otsu Thresholding:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(img_Otsu_plt)

plt.show()
