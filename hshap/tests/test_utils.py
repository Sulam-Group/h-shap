import torch
import numpy as np
from hshap.utils import (
    hshap_features,
    make_masks,
    w,
    shapley_matrix,
    mask2d,
)
from pytest import approx, raises


def test_hshap_features():
    features_ref = np.array(
        [[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]]], dtype=int
    )

    features = hshap_features(4)
    assert np.array_equal(features, features_ref)


def test_make_masks():
    masks_ref = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4],
            [1, 2, 0, 0],
            [1, 0, 3, 0],
            [1, 0, 0, 4],
            [0, 2, 3, 0],
            [0, 2, 0, 4],
            [0, 0, 3, 4],
            [1, 2, 3, 0],
            [1, 2, 0, 4],
            [1, 0, 3, 4],
            [0, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    ).long()

    masks = make_masks(4)
    assert torch.equal(masks, masks_ref)


def test_w():
    gamma = 4

    w0 = w(0, gamma)
    assert w0 == approx(1 / gamma)

    w1 = w(1, gamma)
    assert w1 == approx(1 / (3 * gamma))

    w2 = w(2, gamma)
    assert w2 == approx(1 / (3 * gamma))

    w3 = w(3, gamma)
    assert w3 == approx(1 / gamma)

    assert w0 + 3 * w1 + 3 * w2 + w3 == approx(1)


def test_shapley_matrix():
    gamma = 5
    with raises(NotImplementedError):
        shapley_matrix(gamma)

    gamma = 4
    W = shapley_matrix(gamma)
    F = torch.ones((1, 2 ** gamma))
    phi = torch.matmul(F, W)
    assert torch.allclose(phi, torch.zeros((1, gamma)), atol=1e-06)

    F[0, 0] = 0
    phi = torch.matmul(F, W)
    assert torch.allclose(phi, 0.25 * torch.ones((1, gamma)), atol=1e-06)


def test_mask_single_level():
    x = torch.ones(1, 4, 4)
    background = torch.zeros(1, 4, 4)

    path = torch.tensor([[0, 0, 0, 0]]).long()
    masked_x = mask2d(path, x, background.clone())
    assert torch.equal(masked_x, background)

    path = torch.tensor([[1, 1, 1, 1]]).long()
    masked_x = mask2d(path, x, background.clone())
    print(masked_x)
    assert torch.equal(masked_x, x)


#     path = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]]).long()
#     masked_x = mask2d(path, x, background.clone())
#     assert torch.equal(masked_x, background)

#     path = torch.tensor([[1, 0, 0, 0]]).long()
#     masked_x = mask2d(path, x, background.clone())
#     masked_ref = torch.tensor(
#         [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
#     ).float()
#     assert torch.equal(masked_x, masked_ref)

#     path = torch.tensor([[0, 1, 0, 0]]).long()
#     masked_x = mask2d(path, x, background.clone())
#     masked_ref = torch.tensor(
#         [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
#     ).float()
#     assert torch.equal(masked_x, masked_ref)

#     path = torch.tensor([[0, 0, 1, 0]]).long()
#     masked_x = mask2d(path, x, background.clone())
#     masked_ref = torch.tensor(
#         [[[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]]
#     ).float()
#     assert torch.equal(masked_x, masked_ref)

#     path = torch.tensor([[0, 0, 0, 1]]).long()
#     masked_x = mask2d(path, x, background.clone())
#     masked_ref = torch.tensor(
#         [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]
#     ).float()
#     assert torch.equal(masked_x, masked_ref)


# def test_mask_multiple_levels():
#     x = torch.ones(1, 4, 4)
#     background = torch.zeros(1, 4, 4)

#     path = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]).long()
#     masked_x = mask2d(path, x, background)
#     assert torch.equal(masked_x, background)

#     path = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).long()
#     masked_x = mask2d(path, x, background)
#     masked_ref = torch.tensor(
#         [[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
#     ).float()
#     assert torch.equal(masked_x, masked_ref)

#     path = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 1]]).long()
#     masked_x = mask2d(path, x, background)
#     print(masked_x)
#     masked_ref = torch.tensor(
#         [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
#     ).float()
#     assert torch.equal(masked_x, masked_ref)
