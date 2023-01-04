import os
import glob
import time
import cv2
import numpy as np

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
    miou = np.nanmean(iu[1:])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    P = np.diag(hist) / hist.sum(axis=1)
    R = np.diag(hist) / hist.sum(axis=0)
    F1 = 2 * (P * R) / (P + R)

    return acc, miou, F1[0], F1[1], F1[2], F1[3], F1[4]

gt_path = '/home/data1/jojolee/Tianzhibei2020/Tianzhibei2019OriginalDataset/exp/test_label'
pre_path = '/home/data1/jojolee/Tianzhibei2020/Tianzhibei2019OriginalDataset/exp/Segnet'

gt_path_name = glob.glob(os.path.join(gt_path, '*.png'))
pre_path_name = glob.glob(os.path.join(pre_path, '*.png'))
gt_path_name.sort()
pre_path_name.sort()
gts_all, predictions_all = [], []

start_time = time.time()

numClasses = 5
print(len(gt_path_name))
for i in range(len(gt_path_name)):
    st_time = time.time()
    gt_name = os.path.basename(gt_path_name[i])
    gt = cv2.imread(gt_path_name[i], -1)
    pre = cv2.imread(pre_path_name[i], -1)
    gts_all.append(gt)
    predictions_all.append(pre)

    test_acc, test_miou, test_f1_0, test_f1_1, test_f1_2, test_f1_3, test_f1_4,  = evaluate(
        pre, gt, numClasses)
    ed_time = time.time()
    per_time = ed_time - st_time
    print("gt_name:%s,test_time:%.2f,test_acc:%.4f,test_miou:%.4f,test_0:%.4f,"
          "test_1:%.4f,test_2:%.4f,test_3:%.4f,test_4:%.4f"
          % (gt_name, per_time, test_acc, test_miou, test_f1_0,
             test_f1_1, test_f1_2, test_f1_3, test_f1_4))

all_acc, miou, all_f1_0, all_f1_1, all_f1_2, all_f1_3, all_f1_4, = evaluate(
    predictions_all, gts_all, numClasses)
end_time = time.time()
print("all time:%.0f,all_acc:%.4f,miou:%.4f,0:%.4f,"
      "1:%.4f,2:%.4f,Tree :%.4f,3:%.4f"
      % (end_time - start_time, all_acc, miou, all_f1_0,
         all_f1_1, all_f1_2, all_f1_3, all_f1_4))
