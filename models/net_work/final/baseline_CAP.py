import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        ################################################################################################################
        # 绝对路径
        pretrain_dict = torch.load(
            "/home/liuyuzhe/jiachao_code/seg_exp/project/resnet101-5d3b4d8f.pth")
        ################################################################################################################
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(PyramidPoolingModule, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim, momentum=.95),
            nn.ReLU(inplace=True))
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [self.inconv(x)]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class CAP(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(CAP, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim, momentum=.95),
            nn.ReLU(inplace=True))
        self.features = []
        self.attentions = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))

            self.attentions.append(nn.Sequential(
                nn.Conv2d(in_dim, s * s, kernel_size=1, bias=False),
                nn.BatchNorm2d(s * s, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.attentions = nn.ModuleList(self.attentions)

    def forward(self, x):
        x_d = self.inconv(x)
        # x_size = x.size()
        out = []
        for f, a in zip(self.features, self.attentions):
            fea = f(x)  # [4, 128, s, s]
            att = a(x)  # [4, s*s, 32, 32]
            b, k, h, w = att.size()
            att = att.permute(0, 2, 3, 1)
            att = att.view(b, -1, k)
            fea = fea.view(b, k, -1)
            af = torch.matmul(att, fea)
            af = af.permute(0, 2, 1)
            af = af.view(b, -1, h, w)
            af = af * x_d
            out.append(af)
        out = torch.cat(out, 1)
        return out


class PyramidPoolingModuleWithAtt(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(PyramidPoolingModuleWithAtt, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim, momentum=.95),
            nn.ReLU(inplace=True))
        self.features = []
        self.attentions = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))

            self.attentions.append(nn.Sequential(
                nn.Conv2d(in_dim, s * s, kernel_size=1, bias=False),
                nn.BatchNorm2d(s * s, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.attentions = nn.ModuleList(self.attentions)

    def forward(self, x):
        x_size = x.size()
        out = [self.inconv(x)]
        for f, a in zip(self.features, self.attentions):
            fea = f(x)  # [4, 128, s, s]
            att = a(x)  # [4, s*s, 32, 32]
            b, k, h, w = att.size()
            att = att.permute(0, 2, 3, 1)
            att = att.view(b, -1, k)
            fea = fea.view(b, k, -1)
            af = torch.matmul(att, fea)
            af = af.permute(0, 2, 1)
            af = af.view(b, -1, h, w)
            out.append(af)
        out = torch.cat(out, 1)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)
        # psp module
        # self.pspmodule = PyramidPoolingModule(2048, 256, (1, 2, 3, 6))
        # self.pspmodule = PyramidPoolingModuleWithAtt(2048, 256, (1, 2, 3, 6))



        # self.pspmodule = CAP(2048, 256, (1, 2, 3, 6, 32))
        self.pspmodule = CAP(2048, 256, (5, 5))
        # self.pspmodule = CAP(2048, 256, (4, 4, 4, 4, 32))

        self.relu = nn.ReLU()

        # self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv1 = nn.Conv2d(512, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        # self.conv3 = Conv1x1(2048, 1280)

    def forward(self, input):  # 1,3,512,512
        x, low_level_features = self.resnet_features(input)  # x:1,2048,64,64;low_level_features:1,256,128,128

        # w = self.conv3(x)  # w:1,1280,64,64
        x = self.pspmodule(x)  # x:1,1280,64,64
        # x = w * x
        x = self.conv1(x)  # x:1,256,64,64
        x = self.bn1(x)
        x = self.relu(x)

        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                   int(math.ceil(input.size()[-1] / 4))), mode='bilinear',
                          align_corners=True)  # x:1,256,128,128

        low_level_features = self.conv2(low_level_features)  # low_level_features:1,48,128,128
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        #
        # print(x.shape,low_level_features.shape)
        x = torch.cat((x, low_level_features), dim=1)  # x:1,304,64,64

        x = self.last_conv(x)  # x:1,5,128,128
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # x:1,5,512,512

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnext.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=5, os=16, pretrained=True, _print=True)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
