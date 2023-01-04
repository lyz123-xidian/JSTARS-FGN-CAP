import torch
import os
import numpy as np


from models.compared.CCnet import DeepLabv3_plus as ccnet
from models.compared.segnet import SegNet
from models.compared.pspnet import DeepLabv3_plus as pspnet
from models.compared.seg_hrnet import get_seg_model as hrnet
from models.compared.FCN8S import FCN8s
from models.compared.deeplabv3 import DeepLabv3_plus
from models.compared.ocnet import DeepLabv3_plus as ocnet
from valid.utils import _fast_hist, evaluate_hist
from models.net_work.final.baseline_FGM import DeepLabv3_plus

import glob
from valid.utils import TTA, evaluate
import time
import cv2
import torchvision

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
numClasses = 6
# patch = 1024
# inner = 1024
# stride = 512
scales = [0.75,1.0,1.25]

patch = 512
inner = 512
stride = 256

# 保存模型的地址

# ckptpath = './ckpt_Potsdam/lowsft_attpsp_edge'
ckptpath = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/project/ckpt_addition/add1.0/'
model_path = ckptpath + '/net_26epoch.pth'


# ckptpath = './ckpt_P/res152_egft_appm'
# model_path = ckptpath + '/net_74epoch.pth'

# ckptpath = './ckpt_S_exp/s=6'
# model_path = ckptpath + '/net_96epoch.pth'

# ckptpath = './new_train/v3_os8/V'
# model_path = ckptpath + '/net_52epoch.pth'

# ckptpath = './V_compare/ocnet'
# model_path = ckptpath + '/net_98epoch.pth'

img_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/test/img/'
ground_path = '/home/sd1/liuyuzhe/jiachao_code/seg_exp/data/ori/Vaihingen/test/label/'

# img_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/test/RGB'
# ground_path = '/home/data1/jojolee/seg_exp/data/ori/Potsdam/test/label_no_boundary'

# img_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/train_val/512/img/val'
# ground_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/train_val/512/label/val'

# img_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/1024/img'
# ground_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/1024/label'

# img_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/img'
# ground_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/label'

result_path = ckptpath + '/result/'
result_path_01 = result_path + 'results_01'
result_path_rgb = result_path + 'results_rgb'

# result_path = '/home/data1/jojolee/seg_exp/data/ori/Vaihingen/test/1024/ocnet/'
# result_path_01 = result_path + 'results_01'
# result_path_rgb = result_path + 'results_rgb'

# save results

os.makedirs(result_path_01, exist_ok=True)  # os.mkdir只能创建一级目录
os.makedirs(result_path_rgb, exist_ok=True)  # exist_ok 默认为false，即目录存在时报错，true时不报错


# model = SegNet(numClasses)
# model = pspnet(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = DeepLabv3_plus(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = ccnet(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = ocnet(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = ebgnet(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = hrnet(numClasses,False)
# model = resnet101_attpsp(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = resnet101_aam(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = attaspp_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
model = DeepLabv3_plus(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
model = torch.nn.DataParallel(model)
# model = new_os8(nInputChannels=3, n_classes=numClasses, os=8, _print=False)
# model = DeepLabv3_plus(nInputChannels=3, n_classes=numClasses, os=8, _print=False)
# model = att_aspp(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = lowsft_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = resnet101(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = resnet101_aam_lowsft_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = center_lowsft_edge(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = resnet50(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res50_egftm(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res50_egftm_appm(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res50_appm(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = resnet152(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res152_egftm(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res152_egftm_appm(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res152_appm(nInputChannels=3, n_classes=numClasses, os=16, _print=False)

# model = nn.DataParallel(model)

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model.load_state_dict(model_param)

model = model.cuda()
model.eval()
img_path_name = glob.glob(os.path.join(img_path, '*.tif'))
label_path_name = glob.glob(os.path.join(ground_path, '*.tif'))
img_path_name.sort()
label_path_name.sort()

gts_all, predictions_all = [], []
start_time = time.time()
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
for i in range(len(img_path_name)):
    st_time = time.time()
    img_name = os.path.basename(img_path_name[i])
    img = cv2.imread(img_path_name[i], -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    img = np.array(img, dtype=np.float32)
    img = np.transpose(img, [2, 0, 1]) / 255.0
    img = torch.Tensor(img)
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)  # 1*c*h*w

    # img = fill(np.array(img))
    # img = torch.Tensor(img).cuda()

    label = cv2.imread(label_path_name[i], -1)

    # label = np.int64(np.array(label))  # h*w
    # print(np.unique(label))
    # eval
    with torch.no_grad():
        predictions = TTA(img, model, 1, patch, inner, stride, numClasses, scales)  # N*C*H*W
        # predictions = model(img.cuda())

    # predictions = predictions[:, :, :h, :w]  # N*C*H*W
    predictions = predictions.data.max(1)[1].cpu().numpy()  # N*H*W
    predictions = predictions.squeeze(0)  # H*W(N=1)

    gts_all.append(label)
    predictions_all.append(predictions)

    cv2.imwrite(os.path.join(result_path_01, img_name[:-4] + '.png'), predictions)
    predictions_map = np.zeros((h, w, c), dtype=np.uint8)

    for k in range(6):
        mask = (predictions == k)
        if k == 0:
            predictions_map[mask] = [255, 255, 255]
        if k == 1:
            predictions_map[mask] = [0, 0, 255]
        if k == 2:
            predictions_map[mask] = [0, 255, 255]
        if k == 3:
            predictions_map[mask] = [0, 255, 0]
        if k == 4:
            # predictions_map[mask] = [255, 255, 0]
            predictions_map[mask] = [255, 128, 0]
        if k == 5:
            predictions_map[mask] = [255, 0, 0]

    predictions_map = cv2.cvtColor(predictions_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(result_path_rgb, img_name[:-4] + '.png'), predictions_map)
    hist = _fast_hist(predictions, label, numClasses)
    # test_acc, test_miou, test_f1_0, test_f1_1, test_f1_2, test_f1_3, test_f1_4 = evaluate(
        # predictions, label, numClasses)
    test_acc, test_miou, test_f1_0, test_f1_1, test_f1_2, test_f1_3, test_f1_4, fwiou = evaluate_hist(hist)
    ed_time = time.time()
    per_time = ed_time - st_time
    print("img_name:%s,test_time:%.2f,test_acc:%.4f,test_miou:%.4f,test_0:%.4f,"
          "test_1:%.4f,test_2:%.4f,test_3:%.4f,test_4:%.4f,fwiou:%.4f"
          % (img_name, per_time, test_acc, test_miou, test_f1_0,
             test_f1_1, test_f1_2, test_f1_3, test_f1_4, fwiou))


# hist_all = _fast_hist(predictions_all, gts_all, numClasses)
# all_acc, miou, all_f1_0, all_f1_1, all_f1_2, all_f1_3, all_f1_4 = evaluate(
    # predictions_all, gts_all, numClasses)
all_acc, miou, all_f1_0, all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_fwiou = evaluate(predictions_all, gts_all, numClasses)
end_time = time.time()
print("all time:%.0f,all_acc:%.4f,miou:%.4f,Impervious surfaces:%.4f,"
      "Building:%.4f,Low vegetation:%.4f,Tree :%.4f,Car:%.4f,all_fwiou:%.4f"
      % (end_time - start_time, all_acc, miou, all_f1_0,
         all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_fwiou))

# print("all time:%.0f,all_acc:%.4f,miou:%.4f,Impervious surfaces:%.4f,"
#       "Building:%.4f,Low vegetation:%.4f,Tree :%.4f,Car:%.4f,Clutter/background:%.4f"
#       % (end_time - start_time, all_acc, miou, all_f1_0,
#          all_f1_1, all_f1_2, all_f1_3, all_f1_4, all_f1_5))
import datetime

dtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('./ckpt/result.txt', 'a+') as f:
    f.write("%s,model: %s,time:%.0f,acc:%.4f,miou:%.4f,ImpSurf:%.4f,"
            "Building:%.4f,LowVeg:%.4f,Tree:%.4f,Car:%.4f\n"
            % (dtime, ckptpath.split('/')[-2], end_time - start_time, all_acc, miou, all_f1_0,
               all_f1_1, all_f1_2, all_f1_3, all_f1_4))

