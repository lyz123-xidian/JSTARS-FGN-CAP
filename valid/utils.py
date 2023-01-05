import torch
import numpy as np
import time
import cv2
import torch.nn.functional as F


def fill(img, patch=512, inner=256):
    _, _, h, w = img.shape
    addh, addw = (inner - h % inner) % inner, (inner - w % inner) % inner
    # img = np.pad(img, ((0, 0), (0, 0), (0, addh), (0, addw)), 'constant', constant_values=0)  # 第一次padding，填充为inner的整倍数
    # img = np.pad(img, ((0, 0), (0, 0), (0, addh), (0, addw)), 'reflect')  # 第一次padding，填充为inner的整倍数
    img = np.pad(img, ((0, 0), (0, 0), (0, addh), (0, addw)), 'symmetric')  # 第一次padding，填充为inner的整倍数
    add2 = (patch - inner) // 2
    # img = np.pad(img, ((0, 0), (0, 0), (add2, add2), (add2, add2)), 'constant', constant_values=0)  # 第二次padding，周围填充一圈
    # img = np.pad(img, ((0, 0), (0, 0), (add2, add2), (add2, add2)), 'reflect')  # 第二次padding，周围填充一圈
    img = np.pad(img, ((0, 0), (0, 0), (add2, add2), (add2, add2)), 'symmetric')  # 第二次padding，周围填充一圈
    return img


def pred(img, model, patch=512, inner=256, stride=256, n_classes=2, scales=[1.0]):
    _, _, oh, ow = img.shape
    img = fill(np.array(img), patch=patch, inner=inner)
    img = torch.Tensor(img).cuda()
    n, _, h, w = img.shape
    add2 = (patch - inner) // 2
    prediction = torch.zeros((n, n_classes, h, w)).cuda()
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            patch_img = img[:, :, y:y + patch, x:x + patch]

            patch_pred = torch.zeros((n, n_classes, patch, patch)).cuda()
            for scale in scales:
                input = F.interpolate(patch_img, (int(patch * scale), int(patch * scale)), mode='bilinear')
                output = model(input)
                if isinstance(output, tuple):
                    output = output[0]
                output = torch.softmax(output, dim=1)
                patch_pred += F.interpolate(output, (patch, patch), mode='bilinear')
                del input, output
            inner_pred = patch_pred[:, :, add2:add2 + inner, add2:add2 + inner]  # 取中心inner*inner区域

            prediction[:, :, y:y + inner, x:x + inner] += inner_pred
            # padding后的图patch左上角坐标与原图一一对应且数值一致
    return prediction[:, :, :oh, :ow]


def TTA(img, model, mode=1, patch=512, inner=256, stride=256, n_classes=2, scales=[1.0]):
    prediction = pred(img, model, patch, inner, stride, n_classes, scales)

    # 旋转
    if mode >= 2:
        prediction2 = pred(torch.flip(img, dims=[-1]), model, patch, inner, stride, n_classes, scales)  # 沿y轴翻转
        prediction2 = torch.flip(prediction2, dims=[-1])
        prediction += prediction2
        del prediction2
    # 垂直翻转
    if mode >= 3:
        prediction3 = pred(torch.flip(img, [-2]), model, patch, inner, stride, n_classes, scales)  # 沿x轴翻转
        prediction3 = torch.flip(prediction3, dims=[-2])
        prediction += prediction3
        del prediction3

    if mode >= 4:
        prediction4 = pred(torch.flip(img, [-1, -2]), model, patch, inner, stride, n_classes, scales)  # 沿x,y轴翻转
        prediction4 = torch.flip(prediction4, dims=[-1, -2])
        prediction += prediction4
        del prediction4
    return prediction


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
    freq = hist.sum(axis=1) / hist.sum()
    freq = freq[1:] / freq[1:].sum()
    fwavacc = (freq[freq > 0] * iu[1:][freq > 0]).sum()
    return acc, miou, F1[0], F1[1], F1[2], F1[3], F1[4],fwavacc
    # return acc, miou, iu[0], iu[1], iu[2], iu[3], iu[4], iu[5]
def evaluate_hist(hist):
    acc = np.diag(hist).sum() / hist.sum()
    # acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # mean_iu = np.nanmean(iu)
    mean_iu_noclass0 = np.nanmean(iu[1:])
    freq = hist.sum(axis=1) / hist.sum()
    freq = freq[1:] / freq[1:].sum()
    fwavacc = (freq[freq > 0] * iu[1:][freq > 0]).sum()
    return acc, mean_iu_noclass0, iu[0], iu[1], iu[2], iu[3], iu[4], fwavacc