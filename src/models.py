import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet



OUT_FEATURES = 2

def myModel():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 5),
        nn.ReLU(True),
        nn.MaxPool2d(5, 5),
        nn.Conv2d(16, 16, 3),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(720, 100),
        nn.ReLU(True),
        nn.Linear(100, 15),
        nn.ReLU(True),
        nn.Linear(15, 2)
    )
    return model

def resnet18():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, OUT_FEATURES)
    return model


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class STNLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(STNLayer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))

    def forward(self, x):
        x_ = self.localization(x)
        x_ = x_.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(x_)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class STNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(STNBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.stn = STNLayer()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.stn(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def seNet():
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=OUT_FEATURES)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def stn():
    model = ResNet(STNBasicBlock, [2, 2, 2, 2], num_classes=OUT_FEATURES)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def vgg():
    model = models.vgg19_bn(pretrained=False)
    return model

def mobile():
    return models.mobilenet_v2()

if __name__ == '__main__':
    block = STNLayer(64, )
    batch = torch.ones((10, 64, 50, 70))
    out = block(batch)
    print(out.shape)

