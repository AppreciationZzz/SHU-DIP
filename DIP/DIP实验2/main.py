#数字图像处理-实验2
#注：下文中resize拼错为revise，未全部纠正，了解即可。
#---------------------导入库------------------------------------------
#coding:utf-8
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from PIL import Image

#---------------------用最近邻插值法实现图片放大------------------------------------------
def Nearest(img, bigger_height, bigger_width, channels):
    # 空图预处理
    nearest_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)
    #遍历，range中不包括第二个参数的值，即范围不包括bigger_height与bigger_width
    #序列为0到bigger_height(bigger_width)，共有放大倍数x*bigger_height(bigger_width)个像素
    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            # 按比例增多像素点数量，长宽各自平分为两部分，为后续四舍五入铺垫。
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]
            # round函数：四舍五入
            nearest_row = round(row)
            nearest_col = round(col)
            # 合理化修改, 防越界故自减 1,使得放大3倍后原像素使用次数为2,3,3,3,3,.........3，3，4   （使用次数共计3*原像素个数）
            # 数组尾数下标防越界故自减 1
            if nearest_row == img.shape[0] :
                nearest_row -= 1
            if nearest_col == img.shape[1] :
                nearest_col -= 1
            # 就近赋值
            nearest_img[i][j] = img[nearest_row][nearest_col]

    return nearest_img
#---------------------用双线性插值法实现图片放大------------------------------------------
def Bilinear(img, bigger_height, bigger_width, channels):

    # 空图预处理
    bilinear_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    # 遍历，range中不包括第二个参数的值，即范围不包括bigger_height与bigger_width
    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]

            #int（）去尾法
            row_int = int(row)
            col_int = int(col)
            #计算长宽参数的差值
            u = row - row_int
            v = col - col_int

            # 防止内插时数组下标防越界故自减 1
            if row_int == img.shape[0] -1:
                row_int -= 1
            if col_int == img.shape[1] -1:
                col_int -= 1

            #双线性插值法做线性内插，即做三次单线性插值，四个顶点带权重的公式
            bilinear_img[i][j] = (1 - u) * (1 - v) * img[row_int][col_int] + (1 - u) * v * img[row_int][col_int + 1] \
                                 + u * (1 - v) * img[row_int + 1][col_int] + u * v * img[row_int + 1][col_int + 1]

    return bilinear_img

#---------------------用函数实现图片放大------------------------------------------
#导入图片
img = cv.imread('test1.jpg', cv.IMREAD_COLOR)
#img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  #RGB转换

height, width, channels = img.shape
print("原图像素的长宽大小：")
print(height, width)
bigger_height = height * 2
bigger_width = width * 2
print("处理后图片像素的长宽大小：")
print(bigger_height, bigger_width)

#扩大图片并计算用时,perf_counter()返回当前的计算机系统时间,循环体求平均，减少误差
startTime = time.perf_counter()
for i in range(0, 10):
    Nearest_picture = Nearest(img, bigger_height, bigger_width, channels)
endTime = time.perf_counter()
spend=(endTime-startTime)/10
print("最近邻插值法运行时间为:%f s" %spend)

startTime = time.perf_counter()
for i in range(0, 10):
    Bilinear_picture = Bilinear(img, bigger_height, bigger_width, channels)
endTime = time.perf_counter()
spend=(endTime-startTime)/10
print("双线性插值法运行时间为:%f s" %spend)

# 双三次插值法
# Bicubic_picture = Bicubic(img, bigger_height, bigger_width, channels)

## 用resize（）函数缩放图像，后面的其他程序都是在这一行上改动
#人工设定参数
#dst = cv.resize(img, (400, 300))
startTime = time.perf_counter()
for i in range(0, 1000):
    Revised_picture = cv.resize(img, (0, 0), fx=2, fy=2)
endTime = time.perf_counter()
spend=(endTime-startTime)/1000
print("用resize（）函数运行时间为:%f s" %spend)

# 显示图像
cv.imshow("Nearest_picture: %d x %d" % (Nearest_picture.shape[0], Nearest_picture.shape[1]), Nearest_picture)
cv.imshow("Bilinear_picture: %d x %d" % (Bilinear_picture.shape[0], Bilinear_picture.shape[1]), Bilinear_picture)
# cv.imshow("Bicubic_picture: %d x %d" % (Bicubic_picture.shape[0], Bicubic_picture.shape[1]), Bicubic_picture)
cv.imshow("Resized_picture: %d x %d" % (Revised_picture.shape[0], Revised_picture.shape[1]), Revised_picture)
print("程序运行完毕")

#保存图像
imageRGB1 = cv.cvtColor(Nearest_picture, cv.COLOR_BGR2RGB)
imageRGB2 = cv.cvtColor(Bilinear_picture, cv.COLOR_BGR2RGB)
# imageRGB3 = cv.cvtColor(Bicubic_picture, cv.COLOR_BGR2RGB)
imageRGB4 = cv.cvtColor(Revised_picture, cv.COLOR_BGR2RGB)
picture1 = Image.fromarray(imageRGB1)
picture1.save('Nearest_picture.jpg')
picture2 = Image.fromarray(imageRGB1)
picture2.save('Bilinear_picture.jpg')
# picture3 = Image.fromarray(imageRGB1)
# picture3.save('Bicubic_picture.jpg')
picture4 = Image.fromarray(imageRGB4)
picture4.save('Revised_picture.jpg')
cv.waitKey(0)
cv.destroyAllWindows()

