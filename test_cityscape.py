import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import argparse
from apex import amp

from data.dataset_no_boundary import HyperDataset as Vaigingen
from data.dataset_no_boundary_Potsdam import HyperDataset as Potsdam
from data.dataset_cityscapes import HyperDataset as Cityscapes

# from models.DeepLabv3.deeplab_resnet_nonlocal import DeepLabv3_plus as nl
# from models.backbone.xception_CAP import  DeepLabv3_plus
# from models.backbone.drn_CAP import  DeepLabv3_plus
# from models.backbone.drn_ori import  DeepLabv3_plus
# from models.backbone.xception_ori import DeepLabv3_plus
from models.net_work.final.baseline import DeepLabv3_plus

from utils.utils import AverageMeter, initialize_logger, save_checkpoint, _fast_hist, evaluate_hist

# from utils.lovasz_loss import lovasz_softmax
# from utils.focalloss import focal_loss, focal_loss_my
# from utils.labelsmoothing import Smooth_Critierion
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


args = {
    'ckpt_path': './ckpt_addition/v/cityscapes/resnet101',
    'num_classes': 19,
    'start_epoch': 1, 'iteration': 0,
    'epoch_num': 100, 'batch_size': 8,
    'lr_scheduler': 'poly', 'num_cycle': 4, 'warm_iter': 200,
    'init_lr': 2e-5, 'weight_decay': 1e-4, 'max_iter': 20800, 'power': 0.9,
    'tra_sample_rate': 0.01,
    'record_miou_true': -1
}


def main(args):
    cudnn.benchmark = True
    # load dataset
    print("\nloading dataset ...")


    ckptpath = '/home/liuyuzhe/jiachao_code/seg_exp/project/ckpt_addition/v/cityscapes/'
    model_path = ckptpath + 'resnet101/net_98epoch.pth'


    test_data = Cityscapes(mode='test', aug=True)
    model = DeepLabv3_plus(nInputChannels=3, n_classes=args['num_classes'], os=16, pretrained=True, _print=True)
    model = nn.DataParallel(model)


    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model.load_state_dict(model_param)
    model = model.cuda()
    # train_data = Vaigingen(mode='train', aug=True)
    # train_data = Potsdam(mode='train', aug=True)
    print("Test set samples: ", len(test_data))
    # val_data = Vaigingen(mode='test', aug=False)
    # val_data = Potsdam(mode='test', aug=False)
    # Data Loader (Input Pipeline)
    test_loader = DataLoader(dataset=test_data, batch_size=args['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    # val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'], shuffle=False, num_workers=4,
    #                         pin_memory=True, drop_last=True)

    # args['max_iter'] = len(train_data) // args['batch_size'] * args['epoch_num']
    args['max_iter'] = len(test_loader) * args['epoch_num']

    # Model & Loss
    print("\nbuilding models ...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5 = validate(
        test_loader, model, criterion)
    print(
        "val_loss:%.4f,val_acc:%.4f,val_miou:%.4f,val_f1_0:%.4f,"
        "val_f1_1:%.4f,val_f1_2:%.4f,val_f1_3:%.4f,val_f1_4:%.4f,val_f1_5:%.4f"
        % (val_loss, val_acc, val_miou, val_f1_0,
           val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5))

# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    val_hist = np.zeros((args['num_classes'], args['num_classes']))
    for i, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels.cuda())
            # loss = lovasz_softmax(outputs, labels)
            # loss = focal_loss(outputs, labels)

        # record loss
        losses.update(loss.data)
        predictions = outputs.data
        # predictions1 = predictions.max(1)
        predictions2 = torch.max(predictions, 0)[1]
        # predictions2 = predictions1[1]
        predictions3 = predictions2.cpu().numpy()
        # predictions = outputs.data.max(1)[1].cpu().numpy()
        val_hist += _fast_hist(predictions3.flatten(), labels.cpu().numpy().flatten(), args['num_classes'])

    val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5 = evaluate_hist(val_hist)
    return losses.avg, val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5


if __name__ == '__main__':
    main(args)
