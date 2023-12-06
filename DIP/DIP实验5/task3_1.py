# 数字图像处理-实验5
# 任务3.1-将green绿幕图片中的人物抠出，并融合到tree背景图片中，力求融合结果自然
# ---------------------导入库------------------------------------------
# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ---------------------用双线性插值法实现图片放大------------------------------------------
def Bilinear(img, bigger_height, bigger_width, channels):
    # 空图预处理
    bilinear_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    # 遍历，range中不包括第二个参数的值，即范围不包括bigger_height与bigger_width
    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]

            # int（）去尾法
            row_int = int(row)
            col_int = int(col)
            # 计算长宽参数的差值
            u = row - row_int
            v = col - col_int

            # 防止内插时数组下标防越界故自减 1
            if row_int == img.shape[0] - 1:
                row_int -= 1
            if col_int == img.shape[1] - 1:
                col_int -= 1

            # 双线性插值法做线性内插，即做三次单线性插值，四个顶点带权重的公式
            bilinear_img[i][j] = (1 - u) * (1 - v) * img[row_int][col_int] + (1 - u) * v * img[row_int][col_int + 1] \
                                 + u * (1 - v) * img[row_int + 1][col_int] + u * v * img[row_int + 1][col_int + 1]

    return bilinear_img


# ---------------------设计绿幕抠图融合函数------------------------------------------
def green_screen_extraction(person, background):
    # 读取绿幕背景图像和人物图像
    green_screen_background = background
    person_image = person

    # 转换绿幕背景图像和人物图像为相同的大小
    # shape 属性返回一个包含图像尺寸信息的元组，通常是 (高度, 宽度, 通道数)； resize 函数其中参数表示新图像的宽度和高度
    # 用OpenCv自带函数改图片尺寸（默认使用双线性插值法，可根据参数自行设定）：
    # green_screen_background = cv.resize(green_screen_background, (person_image.shape[1], person_image.shape[0]))
    # 用实验2编写函数改图片尺寸：
    green_screen_background = Bilinear(green_screen_background, person_image.shape[0], person_image.shape[1],
                                       person_image.shape[2])

    # 提取绿幕中的人物部分
    lower_green = np.array([30, 40, 40])  # 绿色的下限
    upper_green = np.array([75, 255, 255])  # 绿色的上限

    # 将绿幕图像转换为HSV颜色空间
    person_image_hsv = cv.cvtColor(person_image, cv.COLOR_BGR2HSV)

    # 创建掩码
    # 这个掩码将标识出HSV图像中在绿幕颜色范围内的像素，形成一个二值掩码，其中绿幕部分为白色（255），非绿幕部分为黑色（0）。
    mask = cv.inRange(person_image_hsv, lower_green, upper_green)

    mask = cv.medianBlur(mask, 5)  # 使用5x5的中值滤波

    # 反转掩码
    # 使白色（255）变成黑色（0），黑色（0）变成白色（255）。
    # 即绿幕部分为黑色（0），非绿幕部分为白色（255）
    mask_inverted = cv.bitwise_not(mask)

    # 从人物图像中提取人物区域
    # 将保留掩码中标记为1的部分，而将标记为0的部分置零，也就是去掉掩码所表示的区域
    person_no_background = cv.bitwise_and(person_image, person_image, mask=mask_inverted)

    # 从背景图中去除即将融合的人物所在的区域
    background_no_green = cv.bitwise_and(green_screen_background, green_screen_background, mask=mask)

    # 合并人物图像和背景图像
    result = cv.add(person_no_background, background_no_green)

    return result


# ---------------------用图片进行测试------------------------------------------
# 导入图片
person_img = cv.imread('green.png')
background_img = cv.imread('tree.jpg')
# 调用抠图函数
result_img = green_screen_extraction(person_img, background_img)
# 保存图片
cv.imwrite('composite_result.jpg', result_img)

# 将融合后图片平滑，使其自然
# 选择要应用高斯模糊的内核大小，可以根据需要调整
kernel_size = (5, 5)
# 使用高斯模糊平滑图像，减少边缘效应
smoothed_image = cv.GaussianBlur(result_img, kernel_size, 0)
# 保存图片
cv.imwrite('Smoothed Image.jpg', smoothed_image)

# 生成最终效果图
result_img_plt = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)
smoothed_image_plt = cv.cvtColor(smoothed_image, cv.COLOR_BGR2RGB)
background_img_plt = cv.cvtColor(background_img, cv.COLOR_BGR2RGB)
person_img_plt = cv.cvtColor(person_img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))  # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(221)  # 创建一个包含多个子图的图形
# 第一个数字 2：表示整个图形分为2行。
# 第二个数字 2：表示整个图形分为2列。
# 第三个数字 1：表示当前子图位于第1个位置。
plt.title('result:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(result_img_plt)

plt.subplot(222)
plt.title('Smoothed result:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(smoothed_image_plt)

plt.subplot(223)
plt.title('background:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(background_img_plt)

plt.subplot(224)
plt.title('person:')
plt.xticks([]), plt.yticks([])  # 不显示x轴y轴数据
plt.imshow(person_img_plt)

plt.show()