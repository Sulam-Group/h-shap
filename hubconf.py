dependencies = ["torch"]

import os
import torch
import torch.nn as nn
from torchvision import models
from demo.RSNA_ICH_detection.model import RSNAHemorrhageNet as _rsnahemorrhagenet


def bbbc041trophozoitenet():
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    dirname = os.path.dirname(__file__)
    state_dict = torch.load(
        os.path.join(dirname, "demo", "BBBC041", "model.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    return model


def rsnahemorrhagenet():
    model = _rsnahemorrhagenet()
    dirname = os.path.dirname(__file__)
    state_dict = torch.load(
        os.path.join(dirname, "demo", "RSNA_ICH_detection", "model.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    return model
