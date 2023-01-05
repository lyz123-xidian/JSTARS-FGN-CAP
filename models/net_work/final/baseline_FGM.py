import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision.models


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

        self.conv_e1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU())

        self.conv_b1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU())

        self.conv_b2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU())

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
        b, e = [], []

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        e1 = self.conv_e1(x)
        e.append(e1)

        x = self.maxpool(x)
        x = self.layer1(x)
        e2 = self.conv_e2(x)
        e.append(e2)

        x = self.layer2(x)
        b1 = self.conv_b1(x)
        b.append(b1)

        x = self.layer3(x)
        b2 = self.conv_b2(x)
        b.append(b2)

        x = self.layer4(x)

        return x, b, e

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = torch.load(
            "/home/sd1/liuyuzhe/jiachao_code/seg_exp/project/resnet101-5d3b4d8f.pth")
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


def ResNet152(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 8, 36, 3], os, pretrained=pretrained)
    return model


def ResNet50(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained)
    return model


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


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, fea, edge):
        if x.shape[-2:] != fea.shape[-2:]:
            x = self.up(x)
        # x = torch.cat((x, fea), dim=1)

        x = x * edge + fea
        x = self.conv_relu(x)
        return x


class FGM(nn.Module):
    def __init__(self, b_channel, e_channel, in_channel):
        super(FGM, self).__init__()

        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(b_channel + in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(e_channel + in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, in_features, body_feature, edge_feature):
        # if x.shape[-2:] != fea.shape[-2:]:
        #     x = self.up(x)
        # x = torch.cat((x, fea), dim=1)

        in_features = F.interpolate(in_features, size=(int(math.ceil(body_feature.size()[-2])),
                                                       int(math.ceil(body_feature.size()[-1]))), mode='bilinear',
                                    align_corners=True)
        f1 = torch.cat((in_features, body_feature), dim=1)
        f1 = self.conv_bn_relu1(f1)
        f1 = self.gap(f1)

        f2 = torch.cat((in_features, edge_feature), dim=1)
        f2 = self.conv_bn_relu1(f2)

        f = in_features * (f1 + f2)
        return f


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


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(2048, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.fgm = FGM(128, 128, 128)

        self.last_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, n_classes, kernel_size=1, stride=1))

        self.aux_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64, n_classes, kernel_size=1, stride=1))

        self.edgeconv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 1, kernel_size=1, stride=1))

    def forward(self, input):  # 1,3,512,512
        x, body, edge = self.resnet_features(input)  # x:1,2048,64,64;low_level_features:1,256,128,128

        segedge = torch.cat((F.interpolate(edge[0], scale_factor=1, mode='bilinear', align_corners=True),
                             F.interpolate(edge[1], scale_factor=2, mode='bilinear', align_corners=True)), 1)

        out_edge = self.edgeconv(segedge)
        out_edge = F.interpolate(out_edge, scale_factor=2, mode='bilinear', align_corners=True)

        segbody = torch.cat((F.interpolate(body[0], scale_factor=4, mode='bilinear', align_corners=True),
                             F.interpolate(body[1], scale_factor=8, mode='bilinear', align_corners=True)), 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        auxseg = segedge + segbody

        out_aux = self.aux_conv(auxseg)
        out_aux = F.interpolate(out_aux, size=input.size()[2:], mode='bilinear', align_corners=True)

        x = self.fgm(x, segbody, segedge)

        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        out_edge1 = torch.sigmoid(out_edge)
        out_edge1 = torch.squeeze(out_edge1)
        # import matplotlib.pyplot as plt
        # plt.imshow(torch.sigmoid(segedge[0, 0, :, :]).cpu())
        # plt.show()
        # plt.imshow(segedge[0, 0, :, :].cpu())
        # plt.show()

        return x, out_aux, out_edge1

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


if __name__ == "__main__":
    from thop import profile

    model = DeepLabv3_plus(nInputChannels=3, n_classes=5, os=16, pretrained=True, _print=True)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    a = torch.zeros(1, 3, 512, 512)
    macs, params = profile(model, inputs=(a,))
    print('FLOPs = ' + str(macs * 2 / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    # a1 = torch.sigmoid(output[:, 1, :] - output[:, 0, :])
    # a2 = torch.softmax(output, dim=1)[:, 1, :]
    # print(a1== a2)  #True
