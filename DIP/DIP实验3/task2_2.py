import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# ——-------------显示直方图--------------------------
# ——----自己实现直方图均衡化---------------------------
img = cv.imread('school.png',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
    #说明：hist,bins = np.histogram(img.flatten(),256,[0,256])
    #256表示直方图的柱子数量，即bins
    #[0, 256]表示指定直方图的范围
    #返回值hist表示每个柱子的频率或计数的数组
    #bins是表示每个柱子的范围的数组
    # img.flatten() 是一个在处理图像数据时常用的操作，通常用于将多维的图像数据数组（例如二维的图像矩阵）转换成一维的数组
cdf = hist.cumsum()  #cdf为累计分布函数，含义是计算直方图的累积分布函数

#自主实现处理直方图均衡化
cdf_m = np.ma.masked_equal(cdf,0)   #将 cdf 中的所有值为零的元素标记为无效（掩码）
                                    # 在直方图均衡化中，如果 cdf 中某些像素值的累积频率为零，这可能会导致除法错误，避免极端映射
                                    # 因此，通过创建一个掩码数组，可以排除这些无效值
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())     #对有效的 cdf 值进行了线性拉伸，
                                                                # 将它们从最小值到最大值的范围映射到 0 到 255 的范围内
cdf = np.ma.filled(cdf_m,0).astype('uint8') #将处理后的 cdf_m 数组转换回普通的 NumPy 数组，并用零填充所有无效值（掩码中的值）
#创建映射表完毕，已经建立像素映射关系
equ_self_image = cdf[img]           #使用经过处理的 cdf 数组来映射原始图像，
                                    #遍历原始图像中的每个像素，将每个像素的值作为索引查找 cdf 数组

#重新计算直方图以及直方图的累积分布函数
hist,bins = np.histogram(equ_self_image.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
    #归一化的累积分布函数：将 cdf 中的每个值都乘以 float(hist.max()),这将使 cdf_normalized 的值范围从 0 到 hist.max()
    #主要作用：帮助曲线图与直方图数据在同一层面，即可视化
plt.plot(cdf_normalized, color = 'b') #绘制曲线图
plt.hist(equ_self_image.flatten(),256,[0,256], color = 'g') #绘制直方图
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left') #添加图例
plt.show()
#比较均衡化效果
res = np.hstack((img,equ_self_image))
cv.imwrite('res_equ_self_image.png',res)