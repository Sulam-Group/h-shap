import numpy as np
import torch
from pytest import approx, raises
from .utils import (
    hshap_features,
    w,
    shapley_matrix,
    mask_features_,
    mask_input_,
    mask_map_,
)

factorial = np.math.factorial


def test_features():
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.bool_,
    )
    expected = np.expand_dims(expected, axis=1)

    gamma = 4
    assert np.array_equal(expected, hshap_features(gamma))


def test_w():
    gamma = 4

    c = 0
    expected = 6 / 24
    assert w(c, gamma) == approx(expected)

    c = 1
    expected = 2 / 24
    assert w(c, gamma) == approx(expected)

    c = 2
    expected = 2 / 24
    assert w(c, gamma) == approx(expected)

    c = 3
    expected = 6 / 24
    assert w(c, gamma) == approx(expected)


def test_shapley_matrix():
    gamma = 3
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    with raises(NotImplementedError, match="Only implemented for gamma = 4"):
        shapley_matrix(gamma, device)

    gamma = 4
    assert shapley_matrix(gamma, device).device == device

    assert shapley_matrix(gamma, device).size() == (2 ** gamma, gamma)

    expected_first_column = torch.tensor(
        [
            -6 / 24,
            6 / 24,
            -2 / 24,
            -2 / 24,
            -2 / 24,
            2 / 24,
            2 / 24,
            2 / 24,
            -2 / 24,
            -2 / 24,
            -2 / 24,
            2 / 24,
            2 / 24,
            2 / 24,
            -6 / 24,
            6 / 24,
        ],
        device=device,
    )
    assert torch.eq(expected_first_column, shapley_matrix(gamma, device)[:, 0]).all()


def test_mask_features_():
    c, h, w = 3, 4, 8
    mask = torch.zeros(5, c, h, w, dtype=torch.bool)
    root_coords = np.array([[0, 0], [h, w]], dtype=np.uint16)

    expected = mask.clone()
    expected[1, :, 0:2, 0:4].fill_(True)
    expected[2, :, 0:2, 4:8].fill_(True)
    expected[3, :, 2:4, 0:4].fill_(True)
    expected[4, :, 2:4, 4:8].fill_(True)

    mask_features_(mask, root_coords)

    assert torch.eq(expected, mask).all()


def test_mask_input_():
    c, h, w = 3, 4, 8
    input = torch.ones(c, h, w)
    background = torch.zeros(c, h, w)
    root_coords = np.array([[0, 0], [h, w]], dtype=np.uint16)

    # first feature
    path_0 = np.array([1, 0, 0, 0])
    expected_0 = background.clone()
    expected_0[:, 0:2, 0:4].fill_(1)

    input_0 = input.clone()
    root_coords_0 = root_coords.copy()
    mask_input_(input_0, path_0, background, root_coords_0)
    assert torch.eq(expected_0, input_0).all()
    assert np.array_equal(root_coords_0, np.array([[0, 0], [2, 4]]))

    # second feature
    path_1 = np.array([0, 1, 0, 0])
    expected_1 = background.clone()
    expected_1[:, 0:2, 4:8].fill_(1)

    input_1 = input.clone()
    root_coords_1 = root_coords.copy()
    mask_input_(input_1, path_1, background, root_coords_1)
    assert torch.eq(expected_1, input_1).all()
    assert np.array_equal(root_coords_1, np.array([[0, 4], [2, 8]]))

    # third feature
    path_2 = np.array([0, 0, 1, 0])
    expected_2 = background.clone()
    expected_2[:, 2:4, 0:4].fill_(1)

    input_2 = input.clone()
    root_coords_2 = root_coords.copy()
    mask_input_(input_2, path_2, background, root_coords_2)
    assert torch.eq(expected_2, input_2).all()
    assert np.array_equal(root_coords_2, np.array([[2, 0], [4, 4]]))

    # fourth feature
    path_3 = np.array([0, 0, 0, 1])
    expected_3 = background.clone()
    expected_3[:, 2:4, 4:8].fill_(1)

    input_3 = input.clone()
    root_coords_3 = root_coords.copy()
    mask_input_(input_3, path_3, background, root_coords_3)
    assert torch.eq(expected_3, input_3).all()
    assert np.array_equal(root_coords_3, np.array([[2, 4], [4, 8]]))


def test_mask_map():
    c, h, w = 3, 4, 8
    map = torch.zeros(c, h, w)
    score = 1
    root_coords = np.array([[0, 0], [h, w]], dtype=np.uint16)

    # first feature
    path_0 = np.array([1, 0, 0, 0])
    expected_0 = map.clone()
    expected_0[:, 0:2, 0:4].fill_(score)
    mask_map_(map, path_0, score, root_coords)

    assert torch.eq(expected_0, map).all()

    # second feature
    path_1 = np.array([0, 1, 0, 0])
    expected_1 = map.clone()
    expected_1[:, 0:2, 4:8].fill_(score)
    mask_map_(map, path_1, score, root_coords)

    assert torch.eq(expected_1, map).all()

    # third feature
    path_2 = np.array([0, 0, 1, 0])
    expected_2 = map.clone()
    expected_2[:, 2:4, 0:4].fill_(score)
    mask_map_(map, path_2, score, root_coords)

    assert torch.eq(expected_2, map).all()

    # fourth feature
    path_3 = np.array([0, 0, 0, 1])
    expected_3 = map.clone()
    expected_3[:, 2:4, 4:8].fill_(score)
    mask_map_(map, path_3, score, root_coords)

    assert torch.eq(expected_3, map).all()
