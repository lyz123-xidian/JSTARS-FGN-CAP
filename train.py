import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time

from apex import amp

from data.dataset_no_boundary import HyperDataset as Vaigingen
# from data.dataset_no_boundary_Potsdam import HyperDataset as Potsdam

# from models.DeepLabv3.deeplab_resnet_nonlocal import DeepLabv3_plus as nl
# from models.backbone.xception_CAP import  DeepLabv3_plus
# from models.backbone.drn_CAP import  DeepLabv3_plus
# from models.backbone.drn_ori import  DeepLabv3_plus
# from models.net_work.final.baseline_CAP import DeepLabv3_plus
from models.net_work.final.baseline_FGM_CAP import DeepLabv3_plus

from utils.utils import AverageMeter, initialize_logger, save_checkpoint,_fast_hist ,evaluate_hist

# from utils.lovasz_loss import lovasz_softmax
# from utils.focalloss import focal_loss, focal_loss_my
# from utils.labelsmoothing import Smooth_Critierion
import random
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

args = {
    'ckpt_path': './ckpt_addition/p/backbone/scale5',
    'num_classes': 6,
    'start_epoch': 1, 'iteration': 0,
    'epoch_num': 60, 'batch_size': 8,
    'lr_scheduler': 'poly', 'num_cycle': 4, 'warm_iter': 200,
    'init_lr': 2e-5, 'weight_decay': 1e-4, 'max_iter': 20800, 'power': 0.9,
    'tra_sample_rate': 0.01,
    'record_miou_true': -1
}


def main(args):
    cudnn.benchmark = True
    # load dataset
    print("\nloading dataset ...")

    # train_data = Vaigingen(mode='train', aug=True)
    train_data = Vaigingen(mode='train', aug=True)
    print("Train set samples: ", len(train_data))
    # val_data = Vaigingen(mode='test', aug=False)
    val_data = Vaigingen(mode='test', aug=False)
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args['batch_size'], shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=True)

    # args['max_iter'] = len(train_data) // args['batch_size'] * args['epoch_num']
    args['max_iter'] = len(train_loader) * args['epoch_num']


    # Model & Loss
    print("\nbuilding models ...")
    model = DeepLabv3_plus(nInputChannels=3, n_classes=args['num_classes'], os=16, pretrained=True, _print=True)

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=args['init_lr'], betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=args['weight_decay'])

    # optimizer = optim.SGD(model.parameters(), lr=args['init_lr'], momentum=0.9, weight_decay=args['weight_decay'])

    if torch.cuda.is_available():
        model.cuda()

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # weight = torch.tensor((0.1, 0.15, 0.35, 0.25, 0.15))
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # criterion = nn.CrossEntropyLoss()
    # criterion = Smooth_Critierion(label_smoothing=0.1)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer
    args['lr'] = args['init_lr']

    # record loss
    checkpoint_path = os.path.join(args['ckpt_path'])
    os.makedirs(checkpoint_path, exist_ok=True)
    # loss_csv = open(os.path.join(checkpoint_path, 'loss.csv'), 'a+')
    log_dir = os.path.join(checkpoint_path, 'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = ''
    # resume_file = checkpoint_path + '/net_49epoch.pth'
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            args['start_epoch'] = checkpoint['epoch'] + 1
            args['iteration'] = checkpoint['iter']
            args['record_miou_true'] = checkpoint['miou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            del checkpoint

    for epoch in range(args['start_epoch'], args['epoch_num'] + 1):
        start_time = time.time()
        tra_loss, tra_acc, tra_miou, tra_f1_0, tra_f1_1, tra_f1_2, tra_f1_3, tra_f1_4, tra_f1_5 = train(
            train_loader, model, criterion, optimizer, epoch, args)
        val_loss, val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5 = validate(
            val_loader, model, criterion)
        # Save model
        if args['record_miou_true'] < val_miou:
            args['record_miou_true'] = val_miou
            save_checkpoint(checkpoint_path, epoch, args['iteration'], args['record_miou_true'], model, optimizer)

        if args['lr_scheduler'] == 'cos' and args['iteration'] % (args['max_iter'] // args['num_cycle']) == 0:
            snap_checkpoint_path = args['ckpt_path'] + '_snapshot'
            os.makedirs(snap_checkpoint_path, exist_ok=True)
            save_checkpoint(snap_checkpoint_path, epoch, args['iteration'], args['record_miou_true'], model,
                            optimizer)

        end_time = time.time()
        epoch_time = end_time - start_time
        # print loss
        print(
            "Epoch[%d],Iter[%d],Time:%.0f,lr:%.9f,tra_loss:%.4f,tra_acc:%.4f,tra_miou:%.4f,tra_f1_0:%.4f,"
            "tra_f1_1:%.4f,tra_f1_2:%.4f,tra_f1_3:%.4f,tra_f1_4:%.4f,tra_f1_5:%.4f"
            % (epoch, args['iteration'], epoch_time, args['lr'], tra_loss, tra_acc, tra_miou, tra_f1_0,
               tra_f1_1, tra_f1_2, tra_f1_3, tra_f1_4, tra_f1_5))
        print(
            "Epoch[%d],Iter[%d],Time:%.0f,lr:%.9f,val_loss:%.4f,val_acc:%.4f,val_miou:%.4f,val_f1_0:%.4f,"
            "val_f1_1:%.4f,val_f1_2:%.4f,val_f1_3:%.4f,val_f1_4:%.4f,val_f1_5:%.4f"
            % (epoch, args['iteration'], epoch_time, args['lr'], val_loss, val_acc, val_miou, val_f1_0,
               val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5))
        # save loss

        logger.info(
            "Epoch[%d],Iter[%d],Time:%.0f,lr:%.9f,tra_loss:%.4f,tra_acc:%.4f,tra_miou:%.4f,tra_f1_0:%.4f,"
            "tra_f1_1:%.4f,tra_f1_2:%.4f,tra_f1_3:%.4f,tra_f1_4:%.4f,tra_f1_5:%.4f"
            % (epoch, args['iteration'], epoch_time, args['lr'], tra_loss, tra_acc, tra_miou, tra_f1_0,
               tra_f1_1, tra_f1_2, tra_f1_3, tra_f1_4, tra_f1_5))
        logger.info(
            "Epoch[%d],Iter[%d],Time:%.0f,lr:%.9f,val_loss:%.4f,val_acc:%.4f,val_miou:%.4f,val_f1_0:%.4f,"
            "val_f1_1:%.4f,val_f1_2:%.4f,val_f1_3:%.4f,val_f1_4:%.4f,val_f1_5:%.4f"
            % (epoch, args['iteration'], epoch_time, args['lr'], val_loss, val_acc, val_miou, val_f1_0,
               val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5))
        # record_loss(loss_csv, epoch, args['iteration'], epoch_time, args['lr'], tra_loss, tra_acc,
        #             tra_miou, tra_f1_0, tra_f1_1, tra_f1_2, tra_f1_3, tra_f1_4,
        #             val_loss, val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5)


# Training

def train(train_data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    losses = AverageMeter()
    tra_hist = np.zeros((args['num_classes'], args['num_classes']))
    for i, (images, labels) in enumerate(train_data_loader):
        start = time.time()
        images = images.cuda()

        # Decaying Learning Rate
        if args['lr_scheduler'] == 'poly':
            lr = poly_lr_scheduler(optimizer, args['init_lr'], args['iteration'], args['max_iter'], args['warm_iter'],
                                   args['power'])
        elif args['lr_scheduler'] == 'cos':
            lr = cosine_lr_scheduler(optimizer, args['init_lr'], args['iteration'], args['max_iter'], args['warm_iter'],
                                     args['num_cycle'],
                                     min_lr=1e-7)
        # Forward + Backward + Optimize
        outputs = model(images)




        loss = criterion(outputs, labels.cuda())
        # loss = lovasz_softmax(outputs, labels
        # loss = focal_loss(outputs, labels)
        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        #  record loss
        losses.update(loss.data)

        predictions = outputs.data.max(1)[1].cpu().numpy()  # output:N*C*H*W
        tra_hist += _fast_hist(predictions.flatten(), labels.numpy().flatten(), args['num_classes'])

        end = time.time()
        ETA = (end - start) * (len(train_data_loader) * (args['epoch_num'] - epoch) + len(train_data_loader) - (
                args['iteration'] % len(train_data_loader))) / 3600
        args['lr'] = lr
        print('[epoch %d],[iter %04d/%04d]:lr = %.9f,train_losses.avg = %.9f,ETA:%.2f'
              % (epoch, args['iteration'] % len(train_data_loader) + 1, len(train_data_loader), args['lr'], losses.avg,
                 ETA))
        args['iteration'] = args['iteration'] + 1

    tra_acc, tra_miou, tra_f1_0, tra_f1_1, tra_f1_2, tra_f1_3, tra_f1_4, tra_f1_5 = evaluate_hist(tra_hist)
    return losses.avg, tra_acc, tra_miou, tra_f1_0, tra_f1_1, tra_f1_2, tra_f1_3, tra_f1_4, tra_f1_5


# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    val_hist = np.zeros((args['num_classes'], args['num_classes']))
    for i, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels.cuda())
            # loss = lovasz_softmax(outputs, labels)
            # loss = focal_loss(outputs, labels)

        # record loss
        losses.update(loss.data)
        predictions = outputs.data.max(1)[1].cpu().numpy()
        val_hist += _fast_hist(predictions.flatten(), labels.cpu().numpy().flatten(), args['num_classes'])

    val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5 = evaluate_hist(val_hist)
    return losses.avg, val_acc, val_miou, val_f1_0, val_f1_1, val_f1_2, val_f1_3, val_f1_4, val_f1_5


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, max_iter=100, warm_iter=10, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion > max_iter:
        return init_lr

    if iteraion <= warm_iter:
        lr = init_lr * (iteraion / warm_iter)
    else:
        lr = init_lr * (1 - (iteraion - warm_iter) / (max_iter - warm_iter)) ** power

    optimizer.param_groups[0]['lr'] = lr

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # print(optimizer.param_groups[0]['lr'])

    return lr


def cosine_lr_scheduler(optimizer, init_lr, iteraion, max_iter, warm_iter, num_cycle=10, min_lr=1e-8):
    if iteraion <= warm_iter:
        lr = init_lr * (iteraion / warm_iter)
    else:
        lr = ((init_lr - min_lr) / 2) * (
                np.cos(np.pi * (np.mod((iteraion - warm_iter), (max_iter - warm_iter) // num_cycle) / (
                        (max_iter - warm_iter) // num_cycle))) + 1) + min_lr

    optimizer.param_groups[0]['lr'] = lr

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    return lr


def compute_weight(label, c):
    number = [0] * c
    for i in range(c):
        number[i] = torch.sum(label == i).item()
    frequency = torch.tensor(number, dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = np.ones(c, dtype=np.float32)
    normHist = frequency / np.sum(frequency)
    for i in range(c):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    classWeights = torch.tensor(classWeights)
    return classWeights


if __name__ == '__main__':
    main(args)
