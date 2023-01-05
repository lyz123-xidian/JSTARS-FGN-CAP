import torch
import torch.nn as nn
import torch.nn.functional as F


class Smooth_Critierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(Smooth_Critierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss()
        else:
            self.criterion = nn.NLLLoss()

    def forward(self, preds, targets):
        n_classes = preds.size(1)
        onehot = F.one_hot(targets.long(), n_classes).permute(0, 3, 1, 2).float()
        onehot = onehot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        if self.label_smoothing > 0:
            loss = self.criterion(self.LogSoftmax(preds), onehot)
            # loss = self.criterion(self.LogSoftmax(preds), onehot.float())
        else:
            loss = self.criterion(self.LogSoftmax(preds), targets)
        return loss


if __name__ == "__main__":
    import numpy as np

    e = 0.1
    n_classes = 2
    crit = Smooth_Critierion(e)
    crit2 = nn.CrossEntropyLoss()

    pred = torch.rand((1, n_classes, 1, 5))  # batchsize,n_class,h,w
    true = torch.from_numpy(np.random.randint(0, n_classes, (1, 1, 5)))
    print(true)
    onehot = F.one_hot(true, n_classes).permute(0, 3, 1, 2).float()
    print(onehot)
    onehot = onehot * (1 - e) + e / n_classes
    # onehot[onehot == 0] = e / n_classes
    print(onehot)
    criterion1 = nn.KLDivLoss()
    criterion2 = nn.NLLLoss()
    log_s = nn.LogSoftmax(dim=1)

    loss1 = criterion1(log_s(pred), onehot.float())
    loss2 = criterion2(log_s(pred), true.long())
    print(onehot.shape,pred.shape,true.shape)

    print(crit(pred, true),crit2(pred, true))
    print(loss1, loss2)
