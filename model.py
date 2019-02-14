import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes, bn=False):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.downStep1 = downStep(1, 64, bn=bn)
        self.downStep2 = downStep(64, 128, bn=bn)
        self.downStep3 = downStep(128, 256, bn=bn)
        self.downStep4 = downStep(256, 512, bn=bn)
        self.downStep5 = downStep(512, 1024, lastLayer=True, bn=bn)
        
        self.upStep1 = upStep(1024, 512, bn=bn)
        self.upStep2 = upStep(512, 256, bn=bn)
        self.upStep3 = upStep(256, 128, bn=bn)
        self.upStep4 = upStep(128, 64, withReLU=False, bn=bn)

        self.conv1 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # todo
        x1 = self.downStep1(x)
        x2 = self.downStep2(x1)
        x3 = self.downStep3(x2)
        x4 = self.downStep4(x3)
        x5 = self.downStep5(x4)

        x = self.up1(x5, x4)
        x = self.up1(x, x3)
        x = self.up1(x, x2)
        x = self.up1(x, x1)

        x = self.conv1(x)

        return x

class downStep(nn.Module):
    def __init__(self, inC, outC, lastLayer=False, bn=False):
        super(downStep, self).__init__()
        # todo
        self.conv_bn = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU())

        self.conv_bn = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # todo
        if self.bn:
            x = self.conv_bn(x)
        else:
            x = self.conv(x)

        if not self.lastLayer:
            x = self.maxpool(x)

        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True, bn=False):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.ReLU())

        self.conv_bn = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.BatchNorm2d(outC),
            nn.Conv2d(outC, outC, 3),
            nn.BatchNorm2d(outC))

        self.conv_relu = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.ReLU(),
            nn.Conv2d(outC, outC, 3),
            nn.ReLU())

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, 3),
            nn.Conv2d(outC, outC, 3))

        self.upsampling = nn.UpsamplingBilinear2d(2)

    def forward(self, x, x_down):
        # todo
        x = self.upsampling(x)
        
        # input is CHW
        p = x_down.size() - x.size()

        x = F.pad(x, (p[3] // 2, p[3] - p[3]//2,
                        p[2] // 2, p[2] - p[2]//2))
        
        x = torch.cat([x_down, x], dim=1)

        if self.withReLU and self.bn: # not last layer, with bn
            x = self.conv_bn_relu(x)

        elif self.withReLU:  # not last layer, without bn
            x = self.conv_relu(x)
        
        elif self.bn: #last layer, with bn
            x = self.conv_bn(x)

        else:  # last layer, without bn
            x = self.conv(x)
        
        return x