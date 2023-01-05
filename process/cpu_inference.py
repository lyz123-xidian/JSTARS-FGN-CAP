import torch
import os
import numpy as np

from models.net_work.resnet101 import DeepLabv3_plus as resnet101
from models.net_work.resnet101_attpsp import DeepLabv3_plus as resnet101_attpsp
from models.net_work.resnet101_attpsp_lowsft_edge import DeepLabv3_plus as attpsp_lowsft_edge
from models.net_work.resnet101_attaspp import DeepLabv3_plus as att_aspp
from models.net_work.resnet101_lowsft_edge import DeepLabv3_plus as lowsft_edge
from models.net_work.resnet101_attaspp_lowsft_edge import DeepLabv3_plus as attaspp_edge
from models.net_work.resnet101_aam import DeepLabv3_plus as resnet101_aam
from models.net_work.resnet101_aam2 import DeepLabv3_plus as resnet101_aam2

import glob
from valid.utils import TTA, evaluate
import time
import cv2
import torchvision

start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
numClasses = 6
patch = 1024
inner = 1024
stride = 512
scales = [1.0]

# 保存模型的地址

# ckptpath = './ckpt_Potsdam/lowsft_attpsp_edge'
ckptpath = '../ckpt_Vaihingen3/aam_lf'
model_path = ckptpath + '/net_84epoch.pth'

img_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/img'
ground_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/label_no_boundary'

# img_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/test/RGB'
# ground_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/test/label_no_boundary'

result_path = ckptpath + '/result/'
result_path_01 = result_path + 'results_01'
result_path_rgb = result_path + 'results_rgb'

# save results

os.makedirs(result_path_01, exist_ok=True)  # os.mkdir只能创建一级目录
os.makedirs(result_path_rgb, exist_ok=True)  # exist_ok 默认为false，即目录存在时报错，true时不报错

loss_csv = open(os.path.join(result_path, 'loss.csv'), 'a+')

#
# model = resnet101_attpsp(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
model = resnet101_aam(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = attaspp_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = attpsp_lowsft_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = att_aspp(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = lowsft_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = resnet101(nInputChannels=3, n_classes=numClasses, os=16, _print=False)

# model = nn.DataParallel(model)

save_point = torch.load(model_path,map_location='cuda')

model_param = save_point['state_dict']
model.load_state_dict(model_param)

model = model.cuda()
model.eval()
img_path_name = glob.glob(os.path.join(img_path, '*.tif'))
label_path_name = glob.glob(os.path.join(ground_path, '*.tif'))
img_path_name.sort()
label_path_name.sort()

gts_all, predictions_all = [], []

normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
for i in range(len(img_path_name)):
    st_time = time.time()
    img_name = os.path.basename(img_path_name[i])

    img = cv2.imread(img_path_name[i], -1)
    label = cv2.imread(label_path_name[i], -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:512, :512, :]
    label = label[:512, :512]
    h, w, c = img.shape

    # normalize = torchvision.transforms.Normalize([0.31172484, 0.32613775, 0.25378257],
    #                                              [0.29395905, 0.27014288, 0.24279141])
    img = np.array(img, dtype=np.float32)
    img = np.transpose(img, [2, 0, 1]) / 255.0
    img = torch.Tensor(img)
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)  # 1*c*h*w

    # img = fill(np.array(img))
    # img = torch.Tensor(img).cuda()


    # label = np.int64(np.array(label))  # h*w
    # print(np.unique(label))
    # eval
    with torch.no_grad():
        # predictions = TTA(img, model, 1, patch, inner, stride, numClasses, scales)  # N*C*H*W
        predictions = model(img.cuda())

    # predictions = predictions[:, :, :h, :w]  # N*C*H*W
    predictions = predictions.data.max(1)[1].cpu().numpy()  # N*H*W
    predictions = predictions.squeeze(0)  # H*W(N=1)

    gts_all.append(label)
    predictions_all.append(predictions)

    # cv2.imwrite(os.path.join(result_path_01, img_name[:-4] + '.png'), predictions)
    # predictions_map = np.zeros((h, w, c), dtype=np.uint8)

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

# print("all time:%.0f,all_acc:%.4f,miou:%.4f,Impervious surfaces:%.4f,"
#       "Building:%.4f,Low vegetation:%.4f,Tree :%.4f,Car:%.4f,Clutter/background:%.4f"
#       % (end_time - start_time, all_acc, miou, all_f1_0,
#          all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_f1_5))
# import datetime

# dtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# with open('./ckpt/result.txt', 'a+') as f:
#     f.write("%s,model: %s,time:%.0f,acc:%.4f,miou:%.4f,ImpSurf:%.4f,"
#             "Building:%.4f,LowVeg:%.4f,Tree:%.4f,Car:%.4f,Clutter:%.4f\n"
#             % (dtime, ckptpath.split('/')[-2], end_time - start_time, all_acc, miou, all_f1_0,
#                all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_f1_5))

# Impervious surfaces (RGB: 255, 255, 255)  0   0.2770868905218727
# Building             (RGB: 0, 0, 255)      1   0.25960237422025795
# Low vegetation      (RGB: 0, 255, 255)    2   0.2122500455724745
# Tree                 (RGB: 0, 255, 0)      3   0.23082855618612744
# Car                  (RGB: 255, 255, 0)    4   0.01220341174787627
# Clutter/background   (RGB: 255, 0, 0)      5   0.008028721751391233
