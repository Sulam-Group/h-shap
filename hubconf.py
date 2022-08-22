dependencies = ["torch"]

import os
import torch
from demo.RSNA_ICH_detection.model import RSNAHemorrhageNet as _rsnahemorrhagenet


def rsnahemorrhagenet(pretrained=False, **kwargs):
    model = _rsnahemorrhagenet()
    if pretrained:
        dirname = os.path.dirname(__file__)
        state_dict = torch.load(
            os.path.join(dirname, "demo", "RSNA_ICH_detection", "model.pt"),
            map_device="cpu",
        )
        model.load_state_dict(state_dict)
    return model
