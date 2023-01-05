import random
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import cv2
from data.augment import augment
import torchvision


class HyperDataset(udata.Dataset):
    def __init__(self, mode='train', aug=True):
        if (mode != 'train') & (mode != 'test'):
            raise Exception("Invalid mode!", mode)
        self.mode = mode
        self.aug = aug
        items = []
        if self.mode == 'train':
            img_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/train/RGB'
            mask_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/train/label_no_boundary'

            img_name = glob.glob(os.path.join(img_path, '*.tif'))
            mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            img_name.sort()
            mask_name.sort()
            for i in range(len(img_name)):
                # print(i)
                # print(os.path.basename(img_name[i]), os.path.basename(mask_name[i]))
                item = (img_name[i], mask_name[i])
                items.append(item)

            # # add augment
            # img_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train_val/512_no_boundary/img/train_add'
            # mask_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/train_val/512_no_boundary/label/train_add'
            # img_name = glob.glob(os.path.join(img_path, '*.tif'))
            # mask_name = glob.glob(os.path.join(mask_path, '*.tif'))
            # img_name.sort()
            # mask_name.sort()
            # for i in range(len(img_name)):
            #     item = (img_name[i], mask_name[i])
            #     items.append(item)

        else:
            img_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/val/RGB'
            mask_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/512/val/label_no_boundary'
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

        return img, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data = HyperDataset(mode='train', aug=True)
    print("Train set samples: ", len(train_data))
    val_data = HyperDataset(mode='test', aug=False)
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=1,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1,
                            pin_memory=False,
                            drop_last=True)  # pin_memory: 锁页内存,设置True表示数据存在计算机内存中（与GPU交互快）,False时计算机内存不足时,会存在虚拟内存中

    for i, (images, labels) in enumerate(val_loader):
        labels = labels
        images = images
        # pre = torch.randn((1, 6, labels.shape[1], labels.shape[2]))
        # print(np.unique(labels))
        # import torch.nn as nn
        #
        # c = nn.CrossEntropyLoss(ignore_index=255)
        # loss = c(pre, labels)
        # print(loss)
