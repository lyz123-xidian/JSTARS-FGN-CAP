import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os


# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def compute_class_weights(histogram, c):
    classWeights = np.ones(c, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(c):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights


def focal_loss_my(input, target):
    '''
    :param input: shape [batch_size,num_classes,H,W] 仅仅经过卷积操作后的输出，并没有经过任何激活函数的作用
    :param target: shape [batch_size,H,W]
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    input = input.permute(0, 2, 3, 1).contiguous().view(-1, c)
    # view要求内存连续，permute和transpose函数会使内存变得不连续，contiguous()将input内存变连续
    target = target.contiguous().view(-1)

    number = [0] * c

    for i in range(c):
        number[i] = torch.sum(target == i).item()

    frequency = torch.tensor(number, dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency, c)
    '''
    根据当前给出的ground truth label计算出每个类别所占据的权重
    '''

    # weights=torch.from_numpy(classWeights).float().cuda()
    weights = torch.from_numpy(classWeights).float()
    print(weights)
    focal_frequency = F.nll_loss(torch.softmax(input, dim=1), target, reduction='none')

    '''
    F.nll_loss(torch.log(F.softmax(inputs, dim=1)，target)的函数功能与F.cross_entropy相同
    可见F.nll_loss中实现了对于target的one-hot encoding编码功能，将其编码成与input shape相同的tensor
    然后与前面那一项（即F.nll_loss输入的第一项）进行 element-wise production
    相当于取出了 log(p_gt)即当前样本点被分类为正确类别的概率
    现在去掉取log的操作，相当于  focal_frequency  shape  [num_samples]
    即取出ground truth类别的概率数值，并取了负号
    '''

    focal_frequency += 1.0  # shape  [num_samples]  1-P（gt_classes）

    focal_frequency = torch.pow(focal_frequency, 2)  # torch.Size([75])
    # print(focal_frequency.shape)
    focal_frequency = focal_frequency.repeat(c, 1)
    # print(focal_frequency.shape)
    '''
    进行repeat操作后，focal_frequency shape [num_classes,num_samples]
    '''
    focal_frequency = focal_frequency.transpose(1, 0)
    loss = F.nll_loss(focal_frequency * (torch.log(F.softmax(input, dim=1))), target, weight=None,
                      reduction='mean')
    return loss


def focal_loss(input, target):
    '''
    :param input: 使用知乎上面大神给出的方案  https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c, h, w = input.size()
    target = target.long()
    inputs = input.permute(0, 2, 3, 1).contiguous().view(-1, c)
    target = target.contiguous().view(-1)
    N = inputs.size(0)  # N=n*h*w
    C = inputs.size(1)

    number = [0] * c

    for i in range(c):
        number[i] = torch.sum(target == i).item()

    frequency = torch.tensor(number, dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency, c)

    weights = torch.from_numpy(classWeights).float().cuda()

    weights = weights[target.view(-1)]  # 这行代码非常重要  weight按照target值扩充为和target维度一致

    gamma = 2
    P = torch.softmax(inputs, dim=1)  # shape [num_samples(n*h*w),num_classes]
    class_mask = inputs.data.new(N, C).fill_(0)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)  # shape [num_samples(n*h*w),num_classes]  one-hot encoding target
    probs = (P * class_mask).sum(1).view(-1, 1)  # shape [num_samples,]
    log_p = probs.log()

    weights = weights.view(n, h, w)
    log_p = log_p.view(n, h, w)
    probs = probs.view(n, h, w)  # resize，否则会超内存

    batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p  # 把每个类别都看做二分类
    # batch_loss = -(torch.pow((1 - probs), gamma)) * log_p  # 把每个类别都看做二分类
    loss = batch_loss.mean()
    return loss


if __name__ == '__main__':
    pred = torch.rand((2, 5, 512, 512))
    y = torch.from_numpy(np.random.randint(0, 5, (2, 512, 512)))
    pred = pred.cuda()
    y = y.cuda()
    loss1 = focal_loss_my(pred, y)
    loss2 = focal_loss(pred, y)

    print('loss1', loss1)
    print('loss2', loss2)
