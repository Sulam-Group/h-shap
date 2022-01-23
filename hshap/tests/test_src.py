import torch
import numpy as np
from hshap.src import Explainer
from hshap.utils import make_masks, hshap_features, shapley_matrix
from torch import Tensor
from pytest import raises


def test_explainer():
    def f(x: Tensor) -> Tensor:
        t = x.flatten(start_dim=1).count_nonzero(dim=1)
        return torch.stack((t == 0, t > 0)).transpose(0, 1).type(torch.float)

    x = torch.zeros(2, 3, 64, 64)
    x[0, :, :2, :2] = 1
    assert torch.all(f(x).eq(torch.Tensor([[0, 1], [1, 0]])))

    background = torch.zeros(3, 64, 64)
    min_size = 2

    hexp = Explainer(
        model=f,
        background=background,
        min_size=min_size,
    )
    assert (
        hexp.model == f
        and torch.all(hexp.background.eq(background))
        and hexp.size == (64, 64)
        and hexp.stop_l == 7  # log(64/2) // log(2) + 2 = 5 + 2 = 7
        and hexp.gamma == 4
        and torch.equal(hexp.masks, make_masks(4))
        and torch.equal(hexp.features, hshap_features(4))
        and torch.equal(hexp.W, shapley_matrix(4))
    )

    explanation_positive_ref = np.zeros((64, 64))
    explanation_positive_ref[:2, :2] = 1
    explanation_positive = hexp.explain(
        x[0], 1, threshold_mode="absolute", binary_map=True
    )
    assert np.array_equal(explanation_positive, explanation_positive_ref)

    with raises(ValueError, match="Could not find any important nodes."):
        hexp.explain(x[1], 1, threshold_mode="absolute")
