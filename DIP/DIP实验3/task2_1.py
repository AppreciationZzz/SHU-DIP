import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# ——-------------显示直方图--------------------------
# ——----OpenCV中对应函数实现直方图均衡化---------------------------
img = cv.imread('school.png',0)
equ_opencv_image = cv.equalizeHist(img) #OpenCV中对应函数实现直方图均衡化
hist,bins = np.histogram(equ_opencv_image.flatten(),256,[0,256])
    #说明：hist,bins = np.histogram(img.flatten(),256,[0,256])
    #256表示直方图的柱子数量，即bins
    #[0, 256]表示指定直方图的范围
    #返回值hist表示每个柱子的频率或计数的数组
    #bins是表示每个柱子的范围的数组
    # flatten() 是一个在处理图像数据时常用的操作，通常用于将多维的图像数据数组（例如二维的图像矩阵）转换成一维的数组
cdf = hist.cumsum()  #cdf为累计分布函数，含义是计算直方图的累积分布函数
cdf_normalized = cdf * float(hist.max()) / cdf.max()
    #归一化的累积分布函数：将 cdf 中的每个值都乘以 float(hist.max()),这将使 cdf_normalized 的值范围从 0 到 hist.max()
    #主要作用：帮助曲线图与直方图数据在同一层面，即可视化
plt.plot(cdf_normalized, color = 'b') #绘制曲线图
plt.hist(equ_opencv_image.flatten(),256,[0,256], color = 'g') #绘制直方图
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left') #添加图例
plt.show()
#比较均衡化效果
res = np.hstack((img,equ_opencv_image))
cv.imwrite('res_equ_opencv_image.png',res)