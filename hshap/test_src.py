import torch
import numpy as np
from pytest import raises
from .src import Explainer, BagExplainer
from .utils import hshap_features, shapley_matrix


def test_explainer():
    def model(x):
        t = x.flatten(start_dim=1).count_nonzero(dim=1)
        return torch.stack((t == 0, t > 0)).transpose(0, 1).type(torch.float)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    background = torch.zeros(3, 64, 64, device=device)

    x = torch.zeros(2, 3, 64, 64, device=device)
    x[1, :, 1, 1] = 1
    x[1, :, 2, 2] = 1
    assert torch.eq(model(x), torch.tensor([[1, 0], [0, 1]], device=device)).all()

    hexp = Explainer(model=model, background=background)
    assert (
        hexp.model == model
        and torch.eq(hexp.background, background).all()
        and hexp.size == (3, 64, 64)
        and hexp.gamma == 4
        and np.array_equal(hshap_features(hexp.gamma), hexp.features)
        and torch.eq(shapley_matrix(hexp.gamma, device), hexp.W).all()
    )

    expected_saliency_map = torch.zeros(1, 64, 64)
    expected_saliency_map[:, 1, 1] = 1
    expected_saliency_map[:, 2, 2] = 1
    assert torch.eq(
        expected_saliency_map,
        hexp.explain(
            x=x[1],
            label=1,
            s=1,
            threshold_mode="absolute",
            threshold=0.0,
            softmax_activation=False,
            batch_size=2,
            binary_map=True,
        ),
    ).all()

    assert torch.eq(
        expected_saliency_map,
        hexp.explain(
            x=x[1],
            label=1,
            s=1,
            threshold_mode="relative",
            threshold=90,
            softmax_activation=False,
            batch_size=2,
            binary_map=True,
        ),
    ).all()

    expected_saliency_map = torch.zeros(1, 64, 64)
    expected_saliency_map[:, :2, :2] = 1
    expected_saliency_map[:, 2:4, 2:4] = 1
    assert torch.eq(
        expected_saliency_map,
        hexp.explain(
            x=x[1],
            label=1,
            s=2,
            threshold_mode="absolute",
            threshold=0.0,
            softmax_activation=False,
            batch_size=2,
            binary_map=True,
        ),
    ).all()

    expected_saliency_map = torch.zeros(1, 64, 64)
    expected_saliency_map[:, 1:3, 1:3] = 1
    assert torch.eq(
        expected_saliency_map,
        hexp.explain(
            x=x[1],
            label=1,
            s=2,
            threshold_mode="absolute",
            threshold=0.0,
            softmax_activation=False,
            batch_size=2,
            binary_map=True,
            roll_row=1,
            roll_column=1,
        ),
    ).all()


def test_bag_explainer():
    def model(b):
        return (
            b.flatten(start_dim=1).count_nonzero(dim=1).count_nonzero(dim=0) > 0
        ) * torch.ones(1, 1, device=b.device)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    out_features = 1

    r = 10
    x0 = torch.zeros(r, 2, 2, device=device)
    assert model(x0) == 0
    x1 = x0.clone()
    x1[0, 0, 0] = 1
    x1[4, 0, 1] = 1
    x1[7, 1, 0] = 1
    x1[-1, -1, -1] = 1
    assert model(x1) == 1

    empty_output = torch.zeros(1, out_features, device=device)
    bag_hexp = BagExplainer(model, empty_output=empty_output)
    assert (
        bag_hexp.model == model
        and bag_hexp.gamma == 2
        and np.array_equal(hshap_features(bag_hexp.gamma), bag_hexp.features)
        and torch.eq(shapley_matrix(bag_hexp.gamma, device), bag_hexp.W).all()
        and torch.eq(empty_output, bag_hexp.empty_output).all()
    )

    with raises(ValueError, match="Could not find any important nodes."):
        bag_hexp.explain(x0, label=0, s=1)

    expected_saliency_map = torch.zeros((r,))
    expected_saliency_map[0] = 1
    expected_saliency_map[4] = 1
    expected_saliency_map[7] = 1
    expected_saliency_map[-1] = 1
    assert torch.eq(
        expected_saliency_map, bag_hexp.explain(x1, label=0, s=1, binary_map=True)
    ).all()
