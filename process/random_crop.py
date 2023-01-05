import cv2
import os
import glob
import numpy as np
from process.augment_c import augment
from tqdm import tqdm

###############################################################################################
# 不能有中文路径

path_img = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/IRRGB'  # src读取地址
path_label = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/label_no_boundary'  # label读取地址

###############################################################################################

size = (512, 512)  # 裁剪的图像尺寸

# 保存路径

save_path_img = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/train_add/IRRGB'
save_path_label = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/train_add/label_no_boundary'

os.makedirs(save_path_img, exist_ok=True)
os.makedirs(save_path_label, exist_ok=True)


def random_crop(img, label, size=size, start=0):
    h = img.shape[0]
    w = img.shape[1]

    cnt = 0
    num = 10
    while cnt < num:
        y = np.random.randint(0, h - size[0] + 1)
        x = np.random.randint(0, w - size[1] + 1)
        cropped0 = img[y:y + size[0], x:x + size[0]]
        cropped1 = label[y:y + size[1], x:x + size[1]]
        cropped0, cropped1 = augment(cropped0, cropped1)

        cv2.imwrite(os.path.join(save_path_img, str(start + cnt) + ".tif"), cropped0)
        cv2.imwrite(os.path.join(save_path_label, str(start + cnt) + ".tif"), cropped1)
        cnt += 1
        # print(cnt)
    return start + num


img_name = glob.glob(os.path.join(path_img, '*.tif'))
mask_name = glob.glob(os.path.join(path_label, '*.tif'))
img_name.sort()
mask_name.sort()
# exist_num=len(os.listdir(save_path_img))
# print(exist_num)
start = 0
for i in tqdm(range(len(img_name))):
    print(img_name[i].split('/')[-1], mask_name[i].split('/')[-1])
    assert img_name[i].split('/')[-1] == mask_name[i].split('/')[-1]
    print(img_name[i].split('/')[-1])
    img = cv2.imread(img_name[i])
    label = cv2.imread(mask_name[i], cv2.IMREAD_GRAYSCALE)
    start = random_crop(img, label, size, start)
