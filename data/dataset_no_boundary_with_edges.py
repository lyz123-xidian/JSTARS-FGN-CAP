import random
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import cv2
from seg_exp.project.data.augment import augment
import torchvision


class HyperDataset(udata.Dataset):
    def __init__(self, mode='train', aug=True):
        if (mode != 'train') & (mode != 'test'):
            raise Exception("Invalid mode!", mode)
        self.mode = mode
        self.aug = aug
        items = []
        if self.mode == 'train':
            img_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/train_val/512_no_boundary/img/train'
            mask_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/train_val/512_no_boundary/label/train'

            img_name = glob.glob(os.path.join(img_path, '*.tif'))
            mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            img_name.sort()
            mask_name.sort()
            for i in range(len(img_name)):
                # print(i)
                # print(os.path.basename(img_name[i]), os.path.basename(mask_name[i]))
                item = (img_name[i], mask_name[i])
                items.append(item)

            # add augment
            img_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/train_val/512_no_boundary/img/train_add'
            mask_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/train_val/512_no_boundary/label/train_add'
            img_name = glob.glob(os.path.join(img_path, '*.tif'))
            mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            img_name.sort()
            mask_name.sort()
            for i in range(len(img_name)):
                item = (img_name[i], mask_name[i])
                items.append(item)

        else:
            img_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/train_val/512_no_boundary/img/val'
            mask_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/train_val/512_no_boundary/label/val'
            img_name = glob.glob(os.path.join(img_path, '*.tif'))
            mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            img_name.sort()
            mask_name.sort()
            for i in range(len(img_name)):
                # print(os.path.basename(img_name[i]), os.path.basename(mask_name[i]))
                item = (img_name[i], mask_name[i])
                items.append(item)
        self.keys = items
        if self.mode == 'train':
            random.shuffle(self.keys)
        else:
            self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        img = cv2.imread(self.keys[index][0])
        label = cv2.imread(self.keys[index][1], cv2.IMREAD_GRAYSCALE)
        if self.aug:
            img, label = augment(img, label)

        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, [2, 0, 1]) / 255.0
        img = torch.Tensor(img)

        img = normalize(img)
        # label[label == 255] = 6
        label = torch.Tensor(label).long()
        edge = torch.zeros_like(label)
        edge[label == 255] = 1

        return img, label, edge


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data = HyperDataset(mode='train', aug=True)
    print("Train set samples: ", len(train_data))
    val_data = HyperDataset(mode='test', aug=False)
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=1,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1,
                            pin_memory=False,
                            drop_last=True)  # pin_memory: ????????????,??????True??????????????????????????????????????????GPU????????????,False???????????????????????????,????????????????????????

    # for i, (images, labels,_) in enumerate(val_loader):
    #     labels = labels
    #     images = images
    #     print(images.shape,labels.shape)
    for i, (images, labels, _) in enumerate(train_loader):
        labels = labels
        images = images
        print(images.shape, labels.shape)
        # pre = torch.randn((1, 6, labels.shape[1], labels.shape[2]))
        # print(np.unique(labels))
        # import torch.nn as nn
        #
        # c = nn.CrossEntropyLoss(ignore_index=255)
        # loss = c(pre, labels)
        # print(loss)
