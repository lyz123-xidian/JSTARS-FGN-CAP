from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import numpy as np
import os
import glob


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, miou, model, optimizer, save_one=True):
    """Save the checkpoint."""
    state = {
        'epoch': epoch,
        'iter': iteration,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    old = glob.glob(model_path + r'/*.pth')
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))
    if save_one and len(old) > 0:
        os.remove(old[0])


def record_loss_binary(loss_csv, epoch, iteration, epoch_time, lr, tra_loss, tra_acc, tra_acc_cls,
                       tra_miou, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                       val_miou, val_fwavacc):
    """ Record many results."""
    loss_csv.write(
        '{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, tra_loss, tra_acc,
                                                             tra_acc_cls,
                                                             tra_miou, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                                                             val_miou, val_fwavacc))
    loss_csv.flush()
    loss_csv.close


def record_loss(loss_csv, epoch, iteration, epoch_time, lr, tra_loss, tra_acc, tra_acc_cls, tra_miou,
                tra_iou0, tra_iou1, tra_iou2, tra_iou3, tra_iou4, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                val_miou, val_iou0, val_iou1, val_iou2, val_iou3, val_iou4, val_fwavacc):
    """ Record many results."""
    loss_csv.write(
        '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time,
                                                                                           lr, tra_loss, tra_acc,
                                                                                           tra_acc_cls,
                                                                                           tra_miou, tra_iou0, tra_iou1,
                                                                                           tra_iou2, tra_iou3, tra_iou4,
                                                                                           tra_fwavacc, val_loss,
                                                                                           val_acc, val_acc_cls,
                                                                                           val_miou, val_iou0, val_iou1,
                                                                                           val_iou2, val_iou3, val_iou4,
                                                                                           val_fwavacc))
    loss_csv.flush()
    loss_csv.close


def record_loss_2020(loss_csv, epoch, iteration, epoch_time, lr, tra_loss, tra_acc, tra_acc_cls, tra_miou,
                     tra_iou0, tra_iou1, tra_iou2, tra_iou3, tra_iou4, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                     val_miou, val_iou0, val_iou1, val_iou2, val_iou3, val_iou4, val_fwavacc):
    """ Record many results."""
    loss_csv.write(
        '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time,
                                                                                           lr, tra_loss, tra_acc,
                                                                                           tra_acc_cls,
                                                                                           tra_miou, tra_iou0, tra_iou1,
                                                                                           tra_iou2, tra_iou3, tra_iou4,
                                                                                           tra_fwavacc, val_loss,
                                                                                           val_acc, val_acc_cls,
                                                                                           val_miou, val_iou0, val_iou1,
                                                                                           val_iou2, val_iou3, val_iou4,
                                                                                           val_fwavacc))
    loss_csv.flush()
    loss_csv.close


def record_loss_binary(loss_csv, epoch, iteration, epoch_time, lr, tra_loss, tra_acc, tra_acc_cls,
                       tra_miou, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                       val_miou, val_fwavacc):
    """ Record many results."""
    loss_csv.write(
        '{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, tra_loss, tra_acc,
                                                             tra_acc_cls,
                                                             tra_miou, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                                                             val_miou, val_fwavacc))
    loss_csv.flush()
    loss_csv.close


def get_reconstruction(input, num_split, dimension, model):
    """As the limited GPU memory split the input."""
    input_split = torch.split(input, int(input.shape[3] / num_split), dim=dimension)
    output_split = []
    for i in range(num_split):
        var_input = Variable(input_split[i].cuda(), volatile=True)
        var_output = model(var_input)
        output_split.append(var_output.data)
        if i == 0:
            output = output_split[i]
        else:
            output = torch.cat((output, output_split[i]), dim=dimension)

    return output


def reconstruction(rgb, model):
    """Output the final reconstructed hyperspectral images."""
    img_res = get_reconstruction(torch.from_numpy(rgb).float(), 1, 3, model)
    img_res = img_res.cpu().numpy() * 4095
    img_res = np.transpose(np.squeeze(img_res))
    img_res_limits = np.minimum(img_res, 4095)
    img_res_limits = np.maximum(img_res_limits, 0)
    return img_res_limits


def rrmse(img_res, img_gt):
    """Calculate the relative RMSE"""
    error = img_res - img_gt
    error_relative = error / img_gt
    rrmse = np.mean((np.sqrt(np.power(error_relative, 2))))
    return rrmse


def _fast_hist(label_pred, label_true, num_classes):
    num_classes = 2 if num_classes == 1 else num_classes
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
    miou = np.nanmean(iu[:])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    P = np.diag(hist) / hist.sum(axis=1)
    R = np.diag(hist) / hist.sum(axis=0)
    F1 = 2 * (P * R) / (P + R)

    return acc, miou, F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]


def evaluate_hist(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # mean_iu = np.nanmean(iu)
    miou = np.nanmean(iu[:])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    P = np.diag(hist) / hist.sum(axis=1)
    R = np.diag(hist) / hist.sum(axis=0)
    F1 = 2 * (P * R) / (P + R)
    return acc, miou, F1[0], F1[1], F1[2], F1[3], F1[4], F1[5]


def evaluate_binary(predictions, gts, num_classes):
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
    return acc, acc_cls, mean_iu_noclass0, fwavacc
