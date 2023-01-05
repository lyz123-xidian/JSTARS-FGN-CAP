import cv2
import os
import glob
import numpy as np
from process.augment_c import augment

###############################################################################################

path_img = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/RGB'  # src读取地址
path_label = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/label_no_boundary'  # label读取地址

###############################################################################################

size = 512  # 裁剪的图像尺寸
stride = 256  # 滑窗步长
cnt = 1

# 保存路径
save_path_img = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/train/RGB'
save_path_label = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/train/label_no_boundary'

os.makedirs(save_path_img, exist_ok=True)
os.makedirs(save_path_label, exist_ok=True)

def crop(img, label, stride=stride, size=size, count=1):
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h, stride):
        if (y + size - 1 > h - 1):
            y = h - size
        for x in range(0, w, stride):
            if (x + size - 1 > w - 1):
                x = w - size
            cropped0 = img[y:y + size, x:x + size]
            cropped1 = label[y:y + size, x:x + size]
            cv2.imwrite(os.path.join(save_path_img, str(count) + ".tif"), cropped0)
            cv2.imwrite(os.path.join(save_path_label, str(count) + ".tif"), cropped1)
            count += 1
            if (x == w - size):
                break
        if (y == h - size):
            break
    return count


img_name = glob.glob(os.path.join(path_img, '*.tif'))
mask_name = glob.glob(os.path.join(path_label, '*.tif'))
img_name.sort()
mask_name.sort()

for i in range(len(img_name)):
    print(img_name[i].split('/')[-1], mask_name[i].split('/')[-1])
    assert img_name[i].split('/')[-1] == mask_name[i].split('/')[-1]

    img = cv2.imread(img_name[i], -1)
    label = cv2.imread(mask_name[i], cv2.IMREAD_GRAYSCALE)
    cnt = crop(img, label, count=cnt)

    print("Processing image No.\t", i + 1, "/", len(img_name))
    print("Shape of src:\t", img.shape)
    print("Shape of label:\t", label.shape)
    print("Num:", cnt - 1, "\tSize:%d" % size, "\tStride:%d" % stride, "\tCrop done")
    print("\n")
