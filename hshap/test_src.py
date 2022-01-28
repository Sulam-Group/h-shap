import torch
import numpy as np
from pytest import raises
from .src import Explainer
from .utils import hshap_features, shapley_matrix


def test_explainer():
    def model(x):
        t = x.flatten(start_dim=1).count_nonzero(dim=1)
        return torch.stack((t == 0, t > 0)).transpose(0, 1).type(torch.float)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    background = torch.zeros(3, 64, 64, device=device)
    min_size = 1

    x = torch.zeros(2, 3, 64, 64, device=device)
    x[1, :, 0, 0] = 1
    x[1, :, 1, 1] = 1
    assert torch.eq(model(x), torch.tensor([[1, 0], [0, 1]], device=device)).all()

    hexp = Explainer(model, background, min_size)
    assert (
        hexp.model == model
        and torch.eq(hexp.background, background).all()
        and hexp.size == (3, 64, 64)
        and hexp.stop_l == 7
        and hexp.gamma == 4
        and np.array_equal(hshap_features(hexp.gamma), hexp.features)
        and torch.eq(shapley_matrix(hexp.gamma, device), hexp.W).all()
    )

    with raises(ValueError, match="Could not find any important nodes."):
        hexp.explain(
            x[0],
            1,
            threshold_mode="absolute",
            threshold=0.0,
            softmax_activation=True,
            batch_size=2,
        )

    expected_saliency_map = torch.zeros(1, 64, 64)
    expected_saliency_map[:, 0, 0] = 1
    expected_saliency_map[:, 1, 1] = 1
    assert torch.eq(
        expected_saliency_map,
        hexp.explain(
            x[1],
            1,
            threshold_mode="absolute",
            threshold=0.0,
            softmax_activation=True,
            batch_size=2,
            binary_map=True,
        ),
    ).all()

    expected_saliency_map = torch.zeros(1, 64, 64)
    expected_saliency_map[:, 0, 0] = 1
    expected_saliency_map[:, 1, 1] = 1
    assert torch.eq(
        expected_saliency_map,
        hexp.explain(
            x[1],
            1,
            threshold_mode="relative",
            threshold=70,
            softmax_activation=True,
            batch_size=2,
            binary_map=True,
        ),
    ).all()
