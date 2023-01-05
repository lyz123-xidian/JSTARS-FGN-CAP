import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import cv2
import glob
import os

img_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/RGB'
img_name = glob.glob(os.path.join(img_path, '*.tif'))
img_name.sort()

hist_ang_all = np.zeros((8,))
hist_mag_all = np.zeros((8,))

print(len(img_name))

for i in range(len(img_name)):
    print('%.2f' %(i/len(img_name)*100))
    img = cv2.imread(img_name[i],-1)
    # img = np.float32(img)/255.

    # img_blur = np.float32(img_blur)/255.

    # 清晰图像的x,y方向的一阶导(梯度)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction ( in degrees )
    # cv2.cartToPolar这个函数计算二维向量(x,y)的幅度梯度和角度梯度
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # bins = angle.max - angle.min
    # bins_blur = angle_blur.max - angle_blur.min

    # bins = np.arange(360)  # np.histogram的默认bins值为10

    hist_mag, bins = np.histogram(mag, bins=8)
    hist_angle, bins = np.histogram(angle, bins=8)
    hist_ang_all += hist_angle
    hist_mag_all += hist_mag
    # print(hist_angle,bins)

    # 计算清晰图像和模糊图像之间的幅度和角度的梯度差

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
# plt.subplot(311), plt.imshow(img), plt.title('ordinary')
plt.subplot(121), plt.title('ord_angle'), plt.bar(center, hist_ang_all, align='center', width=width)
plt.subplot(122), plt.title('ord_mag'), plt.bar(center, hist_mag_all, align='center', width=width)

plt.show()
