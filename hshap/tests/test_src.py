from typing import Callable
from hshap.src import Node, Explainer
from hshap.utils import make_masks, hshap_features
import torch
from torch import Tensor
import numpy as np


def test_node():
    path = np.array([[1, 1, 1, 1]])
    masks = np.array([[1, 0, 0, 0]])

    node = Node(path)

    assert np.array_equal(node.path, path) and node.score == 1

    x = torch.ones(3, 4, 4)
    background = torch.zeros(3, 4, 4)

    masked_x = node.masked_inputs(masks, x, background)
    masked_ref = torch.zeros(x.size())
    masked_ref[:, :2, :2] = 1
    assert torch.all(masked_x.eq(masked_ref))


def test_explainer():
    def f(x: Tensor) -> Tensor:
        t = x.flatten(start_dim=1).count_nonzero(dim=1)
        return torch.stack((t == 0, t > 0)).transpose(0, 1).type(torch.float)

    x = torch.zeros(2, 3, 64, 64)
    x[0, :, :2, :2] = 1
    assert torch.all(f(x).eq(torch.Tensor([[0, 1], [1, 0]])))

    background = torch.zeros(3, 64, 64)
    min_size = 2
    M = 4

    hexp = Explainer(f, background, min_size, M)
    assert (
        hexp.model == f
        and torch.all(hexp.background.eq(background))
        and hexp.size == (64, 64)
        and hexp.stop_l == 6
        and hexp.M == M
        and np.array_equal(hexp.masks, make_masks(4))
        and np.array_equal(hexp.features, hshap_features(4))
    )

    explanation_positive_ref = np.zeros((64, 64))
    explanation_positive_ref[:2, :2] = 1
    explanation_negative_ref = np.zeros((64, 64))
    (explanation_positive, _) = hexp.explain(x[0], 1, threshold_mode="absolute")
    (explanation_negative, _) = hexp.explain(x[1], 1, threshold_mode="absolute")
    assert np.array_equal(
        explanation_positive, explanation_positive_ref
    ) and np.array_equal(explanation_negative, explanation_negative_ref)
