import torch.nn as nn
from torchvision import models


class RSNAHemorrhageNet(nn.Module):
    def __init__(self):
        super(RSNAHemorrhageNet, self).__init__()
        self.n_dim = 128
        self.encoder = self.__encoder__()
        self.classifier = self.__classifier__()

    def __encoder__(self):
        encoder = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        num_features = encoder.fc.in_features
        encoder.fc = nn.Linear(num_features, self.n_dim)
        nn.init.kaiming_normal_(encoder.fc.weight)
        nn.init.constant_(encoder.fc.bias, 0)
        return encoder

    def __classifier__(self):
        return nn.Sequential(nn.Linear(self.n_dim, 1), nn.Sigmoid())

    def forward(self, x):
        H = self.encoder(x)
        x = self.classifier(H)
        return x
