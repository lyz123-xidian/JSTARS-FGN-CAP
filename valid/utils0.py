import torch
import numpy as np
import time


def record_loss(loss_csv, img_name, val_acc, val_acc_cls, val_miou, val_iou1, val_iou2, val_iou3, val_iou4, val_fwavacc):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{}\n'.format(img_name, val_acc, val_acc_cls, val_miou, val_iou1, val_iou2, val_iou3, val_iou4, val_fwavacc))
    loss_csv.flush()
    loss_csv.close


def get_reconstruction_gpu(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cuda()
    # var_input = input
    with torch.no_grad():
        var_output = model(var_input)

    return var_output.cpu()


def get_reconstruction_cpu(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cpu()
    with torch.no_grad():
        start_time = time.time()
        var_output = model(var_input)
        end_time = time.time()

    return end_time-start_time, var_output.cpu()


def copy_patch1(x, y):
    x[:] = y[:]


def copy_patch2(stride, h, x, y):
    x[:, :, :, :-(h % stride)] = (y[:, :, :, :-(h % stride)] + x[:, :, :, :-(h % stride)]) / 2.0
    x[:, :, :, -(h % stride):] = y[:, :, :, -(h % stride):]


def copy_patch3(stride, w, x, y):
    x[:, :, :-(w % stride), :] = (y[:, :, :-(w % stride), :] + x[:, :, :-(w % stride), :]) / 2.0
    x[:, :, -(w % stride):, :] = y[:, :, -(w % stride):, :]


def copy_patch4(stride, w, h, x, y):
    x[:, :, :-(w % stride), :] = (y[:, :, :-(w % stride), :] + x[:, :, :-(w % stride), :]) / 2.0
    x[:, :, -(w % stride):, :-(h % stride)] = (y[:, :, -(w % stride):, :-(h % stride)] + x[:, :, -(w % stride):, :-(h % stride)]) /2.0
    x[:, :, -(w % stride):, -(h % stride):] = y[:, :, -(w % stride):, -(h % stride):]


def reconstruction_patch_image_gpu(rgb, model, patch, stride, classes):
    """Output the final reconstructed hyperspectral images."""
    _, _, w, h = rgb.shape
    rgb = torch.from_numpy(rgb).float()
    temp_hyper = torch.zeros(1, classes, w, h).float()
    # temp_rgb = torch.zeros(1, 3, w, h).float()
    for x in range(w//stride + 1):
        for y in range(h//stride + 1):
            if x < w // stride and y < h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch]
                hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch] = hyper_patch
                copy_patch1(temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch], hyper_patch)
            elif x < w // stride and y == h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, -patch:]
                hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, -patch:] = hyper_patch
                copy_patch2(stride, h, temp_hyper[:, :, x * stride:x * stride + patch, -patch:], hyper_patch)
            elif x == w // stride and y < h // stride:
                rgb_patch = rgb[:, :, -patch:, y * stride:y * stride + patch]
                hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, -patch:, y * stride:y * stride + patch] = hyper_patch
                copy_patch3(stride, w, temp_hyper[:, :, -patch:, y * stride:y * stride + patch], hyper_patch)
            else:
                rgb_patch = rgb[:, :, -patch:, -patch:]
                hyper_patch = get_reconstruction_gpu(rgb_patch, model)
                # temp_hyper[:, :, -patch:, -patch:] = hyper_patch
                copy_patch4(stride, w, h, temp_hyper[:, :, -patch:, -patch:], hyper_patch)

    return temp_hyper.data.squeeze(0)


def reconstruction_patch_image_cpu(rgb, model, patch, stride):
    """Output the final reconstructed hyperspectral images."""
    all_time = 0
    _, _, w, h = rgb.shape
    rgb = torch.from_numpy(rgb).float()
    temp_hyper = torch.zeros(1, 31, w, h).float()
    # temp_rgb = torch.zeros(1, 3, w, h).float()
    for x in range(w//stride + 1):
        for y in range(h//stride + 1):
            if x < w // stride and y < h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch]
                patch_time, hyper_patch = get_reconstruction_cpu(rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch] = hyper_patch
                copy_patch1(temp_hyper[:, :, x * stride:x * stride + patch, y * stride:y * stride + patch], hyper_patch)
            elif x < w // stride and y == h // stride:
                rgb_patch = rgb[:, :, x * stride:x * stride + patch, -patch:]
                patch_time, hyper_patch = get_reconstruction_cpu(rgb_patch, model)
                # temp_hyper[:, :, x * stride:x * stride + patch, -patch:] = hyper_patch
                copy_patch2(stride, h, temp_hyper[:, :, x * stride:x * stride + patch, -patch:], hyper_patch)
            elif x == w // stride and y < h // stride:
                rgb_patch = rgb[:, :, -patch:, y * stride:y * stride + patch]
                patch_time, hyper_patch = get_reconstruction_cpu(rgb_patch, model)
                # temp_hyper[:, :, -patch:, y * stride:y * stride + patch] = hyper_patch
                copy_patch3(stride, w, temp_hyper[:, :, -patch:, y * stride:y * stride + patch], hyper_patch)
            else:
                rgb_patch = rgb[:, :, -patch:, -patch:]
                patch_time, hyper_patch = get_reconstruction_cpu(rgb_patch, model)
                # temp_hyper[:, :, -patch:, -patch:] = hyper_patch
                copy_patch4(stride, w, h, temp_hyper[:, :, -patch:, -patch:], hyper_patch)
            all_time += patch_time

    img_res = temp_hyper.numpy() * 4095
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 4095)
    img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res_limits


def reconstruction_whole_image_gpu(rgb, model):
    """Output the final reconstructed hyperspectral images."""
    all_time, img_res = get_reconstruction_gpu(torch.from_numpy(rgb).float(), model)
    img_res = img_res.cpu().numpy() * 4095
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 4095)
    img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res_limits


def reconstruction_whole_image_cpu(rgb, model):
    """Output the final reconstructed hyperspectral images."""
    all_time, img_res = get_reconstruction_cpu(torch.from_numpy(rgb).float(), model)
    img_res = img_res.cpu().numpy() * 4095
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 4095)
    img_res_limits = np.maximum(img_res_limits, 0)
    return all_time, img_res_limits


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    num_classes = 2 if num_classes == 1 else num_classes
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # mean_iu = np.nanmean(iu)
    mean_iu_noclass0 = np.nanmean(iu[1:])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu_noclass0, iu[1],  fwavacc

