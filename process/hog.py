import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import cv2


# img = cv2.imread('/home/data1/jojolee/seg_exp/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',-1)
# img = cv2.imread('/home/data1/jojolee/seg_exp/data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png',-1)
img = cv2.imread('/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/RGB/top_potsdam_2_10.tif',-1)
# img = np.float32(img)/255.

# img_blur = np.float32(img_blur)/255.

# 清晰图像的x,y方向的一阶导(梯度)
gx_ord = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy_ord = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)



# Calculate gradient magnitude and direction ( in degrees )
# cv2.cartToPolar这个函数计算二维向量(x,y)的幅度梯度和角度梯度
mag_ord, angle_ord = cv2.cartToPolar(gx_ord, gy_ord, angleInDegrees=True)


# bins_ord = angle_ord.max - angle_ord.min
# bins_blur = angle_blur.max - angle_blur.min

# bins = np.arange(360)  # np.histogram的默认bins值为10


hist_ord_mag, bins = np.histogram(mag_ord, bins=8,range=(0,360))
hist_ord_angle, bins = np.histogram(angle_ord, bins=8,range=(0,360))

# print(hist_ord_angle.shape,bins)

# 计算清晰图像和模糊图像之间的幅度和角度的梯度差

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(20, 20))
# 设置刻度字体大小
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)



plt.subplot(131), plt.imshow(img), plt.title('img',fontsize=20)

plt.subplot(132), plt.imshow(gx_ord), plt.title('gx',fontsize=20)
plt.subplot(133), plt.imshow(gy_ord), plt.title('gy',fontsize=20)

# plt.subplot(312), plt.title('angle'), plt.bar(center, hist_ord_angle, align='center', width=width)
#
# plt.subplot(313), plt.title('mag'), plt.bar(center, hist_ord_mag, align='center', width=width)

plt.show()