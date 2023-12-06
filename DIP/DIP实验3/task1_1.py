#-------------导入库---------------------------------------
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

# ——-------------伽马变换函数，默认gamma值为1--------------------------
#不使用查找表
def adjust_gamma(image, gamma=1):
    r_img=image/255.0
    #根据幂律变换形式所得公式
    corrected_image=np.power( r_img, gamma)*255
    corrected_image=np.round(corrected_image)
    brighter_image = np.array(corrected_image, dtype=np.uint8)
    return brighter_image

#---------导入图片--------------------------
img = cv.imread('light.tif', 0)

#----------设置图样显示框架--------------------------
#设置
plt.figure(figsize=(8,8))   # 创建了一个新的图形窗口，并指定了图形的大小为8x8英寸
plt.subplot(131)    #创建一个包含多个子图的图形
                    # 第一个数字 1：表示整个图形分为1行。
                    # 第二个数字 3：表示整个图形分为3列。
                    # 第三个数字 1：表示当前子图位于第1个位置。
plt.title('gamma = 1')
plt.imshow(img, cmap = 'gray')   # 原图

#图像进行伽马变换
startTime = time.perf_counter()
for i in range(0, 100):
    img_gamma = adjust_gamma(img, gamma=5)
endTime = time.perf_counter()
spend=(endTime-startTime)/100
print("不使用查找表进行伽马变换用时:%f s" %spend)

plt.subplot(132)
plt.title('gamma = 5 > 1')
plt.imshow(img_gamma, cmap="gray")

# gamma小于1，变暗
img_gamma = adjust_gamma(img, gamma=0.4)
plt.subplot(133)
plt.title('gamma =0.4 < 1')
plt.imshow(img_gamma, cmap="gray")
plt.show()
