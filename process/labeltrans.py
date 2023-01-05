import os
import cv2
import glob
import numpy as np

labelpath = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/1024/label'
savepath = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/1024/label_color'

os.makedirs(savepath, exist_ok=True)

labelname = glob.glob(os.path.join(labelpath, '*.tif'))
labelname.sort()
for i in range(len(labelname)):
    print(labelname[i])
    label = cv2.imread(labelname[i], -1)
    h, w = label.shape
    labelmap = np.zeros((h, w, 3), dtype=np.uint8)
    labelmap[label == 0] = [255, 255, 255]
    labelmap[label == 1] = [0, 0, 255]
    labelmap[label == 2] = [0, 255, 255]
    labelmap[label == 3] = [0, 255, 0]
    labelmap[label == 4] = [255, 128, 0]
    labelmap[label == 5] = [255, 0, 0]
    labelmap = cv2.cvtColor(labelmap, cv2.COLOR_BGR2RGB)
    savename = os.path.join(savepath, os.path.basename(labelname[i]))

    cv2.imwrite(savename, labelmap)
