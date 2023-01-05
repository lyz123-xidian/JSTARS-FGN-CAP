import matplotlib.pyplot as plt

epoch, tra_loss, val_loss, tra_miou, val_miou, lr = [], [], [], [], [], []
# ckptpath = '/home/data1/jojolee/seg_exp/project/train_on_cityscapes/ckpt/deeplabv3+_correct_augmeent_weighted'
# ckptpath = '/home/data1/jojolee/seg_exp/project/train_on_cityscapes/ckpt/deeplabv3+'
ckptpath = '/home/data1/jojolee/seg_exp/project/ckpt_V/res50_appm'
s = 0
k = 0
for i, line in enumerate(open(ckptpath + "/train.log")):
    list = line.split(',')
    print(list)
    x = int(list[0].split('[')[1][:-1])
    loss = float(list[4][-6:])
    miou = float(list[6][-6:])
    learing_rate = list[3][3:]
    if s == 0:
        while k < len(learing_rate):
            if learing_rate[k] not in ['0', '.']:
                k -= 2
                s = 10 ** k
                break
            k += 1
    # print(s, k)
    learing_rate = float(learing_rate) * s

    if not i % 2:
        epoch.append(x)
        lr.append(learing_rate)
    if list[4][:3] == 'tra':
        tra_loss.append(loss)
        tra_miou.append(miou)
    else:
        val_loss.append(loss)
        val_miou.append(miou)

print(epoch)
print(tra_loss)
print(val_loss)
# print(tra_miou)
print(val_miou)
print(lr)

plt.plot(epoch, tra_loss, ls="-", lw=2, label='tra_loss')
plt.plot(epoch, val_loss, ls="-", lw=2, label='val_loss')
# plt.plot(epoch, tra_miou, ls="-", lw=2, label='tra_miou')
plt.plot(epoch, val_miou, ls="-", lw=2, label='val_miou')
plt.plot(epoch, lr, ls="-", lw=2, label='lr_e-{}'.format(k))

plt.xlabel('epoch')
plt.ylabel('loss/miou/lr')

# plt.title('loss')

plt.legend()

plt.savefig(ckptpath + '/loss.png')
plt.show()
