#-------------导入库---------------------------------------
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

# ——-------------伽马变换函数，默认gamma值为1--------------------------
#使用查找表
def create_gamma_lut(gamma):
    # 创建查找表
    lut = np.arange(256, dtype=np.uint8)

    # 对查找表中的每个像素值进行幂律变换
    lut = np.power(lut / 255.0, gamma) * 255.0  # np.arange(256) 生成一个从0到255的一维数组序列

    # 将查找表的数据类型转换为无符号整数类型
    lut = np.round(lut).astype(np.uint8)

    return lut

#---------导入图片--------------------------
img = cv.imread('light.tif', 0)

#----------设置图样显示框架--------------------------
#设置
plt.figure(figsize=(8,8))
plt.subplot(131)
plt.title('gamma = 1')
plt.imshow(img, cmap = 'gray')   # 原图

lut = create_gamma_lut(gamma=5) #生成查找表

#图像进行伽马变换
startTime = time.perf_counter()
for i in range(0, 100):
    # 将查找表应用到输入图像上，生成调整后的图像
    # brighter_image = cv.LUT(img, lut)    #用opencv自带的查找表映射函数，用时数量级为等效方法的1/10
    brighter_image = lut[img]  #等效映射效果
endTime = time.perf_counter()
spend=(endTime-startTime)/100
print("使用查找表进行伽马变换用时:%f s" %spend)
plt.subplot(132)
plt.title('gamma = 5 > 1')
plt.imshow(brighter_image, cmap="gray")

# gamma小于1，变暗
lut = create_gamma_lut(gamma=0.4)
brighter_image = cv.LUT(img, lut)   #将查找表应用到输入图像上，生成调整后的图像
plt.subplot(133)
plt.title('gamma =0.4 < 1')
plt.imshow(brighter_image, cmap="gray")
plt.show()
