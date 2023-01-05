import torch
import os
import numpy as np

from models.net_work.final.baseline import DeepLabv3_plus
import glob
import time
import cv2
import torchvision
import torch.nn.functional as F


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

    return acc, iu, miou


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
numClasses = 19

scales = [0.5, 0.75, 1., 1.5, 2.0]

patch = [512, 1024]
stride = [256, 512]

# save results
# result_path = ckptpath + '/result/'
# result_path_01 = result_path + 'results_01'
# result_path_rgb = result_path + 'results_rgb'

# os.makedirs(result_path_01, exist_ok=True)  # os.mkdir只能创建一级目录
# os.makedirs(result_path_rgb, exist_ok=True)  # exist_ok 默认为false，即目录存在时报错，true时不报错

# model = new_2(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = res101_r_ocr(nInputChannels=3, n_classes=numClasses, os=16, _print=False,num_R=2)
model = res101_ocr(nInputChannels=3, n_classes=numClasses, os=16, _print=False)
# model = nn.DataParallel(model)

model_path = '/home/data1/jojolee/seg_exp/project/train_on_cityscapes/ckpt/res101+ocr/net_232epoch.pth'
save_point = torch.load(model_path)
model_param = save_point['state_dict']
model.load_state_dict(model_param)

model = model.cuda()
model.eval()

gts_all, predictions_all = [], []
start_time = time.time()
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

st_time = time.time()

txt = '/home/data1/jojolee/seg_exp/data/cityscapes/val_multitask.txt'
for line in open(txt):
    line = line.split()
    img_name = line[0]
    mask_name = line[-1]
    root = '/home/data1/jojolee/seg_exp/data/cityscapes/'
    img = cv2.imread(root + img_name)
    label = cv2.imread(root + mask_name, cv2.IMREAD_GRAYSCALE)
    # print(np.unique(label))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = np.transpose(img, [2, 0, 1]) / 255.0
    img = torch.Tensor(img).cuda()
    img = normalize(img)
    img = torch.unsqueeze(img, dim=0)  # 1*c*h*w
    batch, _, h, w = img.shape

    predictions = torch.zeros([1, numClasses, h, w]).cuda()
    with torch.no_grad():
        for s in scales:
            new_img = F.interpolate(img, (int(h * s), int(w * s)), mode='bilinear')
            _, _, new_h, new_w = new_img.shape

            rows = np.int(np.ceil(1.0 * (new_h - patch[0]) / stride[0])) + 1
            cols = np.int(np.ceil(1.0 * (new_w - patch[1]) / stride[1])) + 1

            preds = torch.zeros([1, numClasses, new_h, new_w]).cuda()
            count = torch.zeros([1, 1, new_h, new_w]).cuda()
            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride[0]
                    w0 = c * stride[1]
                    h1 = min(h0 + patch[0], new_h)
                    w1 = min(w0 + patch[1], new_w)
                    h0 = max(int(h1 - patch[0]), 0)
                    w0 = max(int(w1 - patch[1]), 0)  # h0,w0限制上限
                    crop_img = new_img[:, :, h0:h1, w0:w1]

                    pred = model(crop_img)

                    if isinstance(pred, tuple) or isinstance(pred, list):
                        pred = model(crop_img)[-1]
                        pred += torch.flip(model(torch.flip(crop_img, dims=[-1]))[-1], dims=[-1])
                    else:
                        pred += torch.flip(model(torch.flip(crop_img, dims=[-1])), dims=[-1])

                    preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                    count[:, :, h0:h1, w0:w1] += 1

            preds = preds / count
            preds = F.interpolate(preds, (h, w), mode='bilinear')
            preds = torch.softmax(preds, dim=1)
            predictions += preds

            del new_img, preds

    # predictions = predictions[:, :, :h, :w]  # N*C*H*W
    predictions = predictions.data.max(1)[1].cpu().numpy()  # N*H*W
    predictions = predictions.squeeze(0)  # H*W(N=1)

    gts_all.append(label)
    predictions_all.append(predictions)

    # cv2.imwrite(os.path.join(result_path_01, img_name[:-4] + '.png'), predictions)
    # predictions_map = np.zeros((h, w, c), dtype=np.uint8)
    #
    # for k in range(6):
    #     mask = (predictions == k)
    #     if k == 0:
    #         predictions_map[mask] = [255, 255, 255]
    #     if k == 1:
    #         predictions_map[mask] = [0, 0, 255]
    #     if k == 2:
    #         predictions_map[mask] = [0, 255, 255]
    #     if k == 3:
    #         predictions_map[mask] = [0, 255, 0]
    #     if k == 4:
    #         # predictions_map[mask] = [255, 255, 0]
    #         predictions_map[mask] = [255, 128, 0]
    #     if k == 5:
    #         predictions_map[mask] = [255, 0, 0]
    #
    # predictions_map = cv2.cvtColor(predictions_map, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(result_path_rgb, img_name[:-4] + '.png'), predictions_map)

    pre_acc, pre_iou, pre_miou = evaluate(predictions, label, numClasses)

    ed_time = time.time()
    per_time = ed_time - st_time
    print("img_name:%s,test_time:%.2f,test_acc:%.4f,test_miou:%.4f"
          % (img_name, per_time, pre_acc, pre_miou))

all_acc, all_iou, all_miou = evaluate(predictions_all, gts_all, numClasses)

end_time = time.time()
print("all time:%.0f,all_acc:%.4f,miou:%.4f"
      % (end_time - start_time, all_acc, all_miou))
print('iou:', all_iou)
