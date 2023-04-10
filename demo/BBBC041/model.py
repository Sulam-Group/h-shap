import torch.nn as nn
from torchvision import models


class BBBC041TrophozoiteNet(nn.Module):
    def __init__(self):
        super(BBBC041TrophozoiteNet, self).__init__()
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 2)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)
