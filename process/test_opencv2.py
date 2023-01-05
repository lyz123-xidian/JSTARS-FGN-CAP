import cv2
import torch

import os
import numpy as np

import glob
import time
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F

def _fast_hist(label_pred, label_true, num_classes):  # rows true, cols pred
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    num_classes = 2 if num_classes == 1 else num_classes
    hist = np.zeros((num_classes, num_classes))
    if isinstance(predictions, list):
        for lp, lt in zip(predictions, gts):
            hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    else:
        hist += _fast_hist(predictions.flatten(), gts.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # mean_iu = np.nanmean(iu)
    miou = np.nanmean(iu[:-1])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    P = np.diag(hist) / hist.sum(axis=1)
    R = np.diag(hist) / hist.sum(axis=0)
    F1 = 2 * (P * R) / (P + R)

    return acc, miou, F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]
    # return acc, miou, iu[0], iu[1], iu[2], iu[3], iu[4], iu[5]

img_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/img'
ground_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/label_no_boundary'
numClasses = 6
# save results

img_path_name = glob.glob(os.path.join(img_path, '*.tif'))
label_path_name = glob.glob(os.path.join(ground_path, '*.tif'))
img_path_name.sort()
label_path_name.sort()

gts_all, predictions_all = [], []
start_time = time.time()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

net = cv2.dnn.readNetFromONNX('network.proto')
for i in range(len(img_path_name)):
    st_time = time.time()
    img_name = os.path.basename(img_path_name[i])
    img = cv2.imread(img_path_name[i], -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img=cv2.dnn.blobFromImage(img)
    img=img[:512,:512,:]
    h, w, c = img.shape

    img = np.array(img, dtype=np.float32)/ 255.0
    img = (img - mean) / std
    img = np.transpose(img, [2, 0, 1])

    # img = torch.Tensor(img)
    # img = torch.unsqueeze(img, dim=0)  # 1*c*h*w

    img = np.expand_dims(img,axis=0)  # 1*c*h*w

    label = cv2.imread(label_path_name[i], -1)
    label=label[:512,:512]

    net.setInput(img)
    predictions = net.forward()

    # predictions = predictions.data.max(1)[1].cpu().numpy()  # N*H*W
    predictions = predictions.argmax(1)
    predictions = predictions.squeeze(0)  # H*W(N=1)

    gts_all.append(label)
    predictions_all.append(predictions)

    test_acc, test_miou, test_f1_0, test_f1_1, test_f1_2, test_f1_3, test_f1_4, test_f1_5, = evaluate(
        predictions, label, numClasses)
    ed_time = time.time()
    per_time = ed_time - st_time
    print("img_name:%s,test_time:%.2f,test_acc:%.4f,test_miou:%.4f,test_0:%.4f,"
          "test_1:%.4f,test_2:%.4f,test_3:%.4f,test_4:%.4f,test_5:%.4f"
          % (img_name, per_time, test_acc, test_miou, test_f1_0,
             test_f1_1, test_f1_2, test_f1_3, test_f1_4, test_f1_5))

all_acc, miou, all_f1_0, all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_f1_5 = evaluate(
    predictions_all, gts_all, numClasses)
end_time = time.time()
print("all time:%.0f,all_acc:%.4f,miou:%.4f,Impervious surfaces:%.4f,"
      "Building:%.4f,Low vegetation:%.4f,Tree :%.4f,Car:%.4f,Clutter/background:%.4f"
      % (end_time - start_time, all_acc, miou, all_f1_0,
         all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_f1_5))


# Runs the forward pass to get output of the output layers

