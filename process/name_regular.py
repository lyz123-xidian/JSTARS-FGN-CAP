import os
import glob
import cv2

path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train/RGB'
name = glob.glob(os.path.join(path, '*.tif'))
for na in name:
    label = cv2.imread(na, -1)
    n = os.path.basename(na)
    print(n)
    t = n[:-8]
    newn = t + '.tif'
    print(newn)
    cv2.imwrite(os.path.join(path, newn), label)
    os.remove(na)
