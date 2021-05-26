import torch
from torch import nn
from torchvision import models


class net_conv1d(nn.Module):
    def __init__(self):
        super(net_conv1d, self).__init__()
        self.conv1 = nn.Conv1d(5, 5, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(5, 5, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(5)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(5, 5, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class net_conv3d(nn.Module):
    def __init__(self):
        super(net_conv3d, self).__init__()
        self.conv1 = nn.Conv3d(5, 5, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(5, 5, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(5)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(5, 5, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class vgg16_bn(nn.Module):
    def __init__(self):
        super(vgg16_bn, self).__init__()
        self.features = models.vgg16_bn(pretrained=False).features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = vgg16_bn()
    random_input2d = torch.randn((1, 3, 32, 32))
    print(net(random_input2d).shape)

    net = net_conv1d()
    random_input1d = torch.randn((1, 5, 32))
    print(net(random_input1d).shape)

    net = net_conv3d()
    random_input3d = torch.randn((1, 5, 32, 32, 32))
    print(net(random_input3d).shape)