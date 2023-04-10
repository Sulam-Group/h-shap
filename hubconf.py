dependencies = ["torch"]

import os
import torch
from demo.BBBC041.model import BBBC041TrophozoiteNet as _bbbc041trophozoitenet
from demo.RSNA_ICH_detection.model import RSNAHemorrhageNet as _rsnahemorrhagenet


def bbbc041trophozoitenet():
    model = _bbbc041trophozoitenet()
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
