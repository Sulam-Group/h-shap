from hshap.utils import (
    enumerate_batches,
    hshap_features,
    make_masks,
    mask2d,
    mask2str,
    str2mask,
    shapley_phi,
    children_scores,
)
import numpy as np
import torch


def test_enumerate_bathces():
    coll = list(range(12))
    batches_ref = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]

    for i, batch in enumerate_batches(coll, batch_size=5):
        assert batch == batches_ref[i]
    assert i == 2


def test_hshap_features():
    features_ref = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    features = hshap_features(4)
    assert (features_ref == features).all()


def test_make_masks():
    masks_ref = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
        ]
    )

    masks = make_masks(4)
    assert sum(
        [sum([(mask == ref).all() for ref in masks_ref]) for mask in masks]
    ) == len(masks_ref)


def test_mask_single_level():
    x = torch.ones(1, 4, 4)
    background = torch.zeros(1, 4, 4)

    path = np.array([])
    masked_x = mask2d(path, x, background.clone())
    assert torch.all(masked_x.eq(x))

    path = np.array([[0, 0, 0, 0]])
    masked_x = mask2d(path, x, background.clone())
    assert torch.all(masked_x.eq(background))

    path = np.array([[1, 1, 1, 1]])
    masked_x = mask2d(path, x, background.clone())
    assert torch.all(masked_x.eq(x))

    path = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    masked_x = mask2d(path, x, background.clone())
    assert torch.all(masked_x.eq(background))

    path = np.array([[1, 0, 0, 0]])
    masked_x = mask2d(path, x, background.clone())
    masked_ref = torch.tensor(
        [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    )
    assert torch.all(masked_x.eq(masked_ref))

    path = np.array([[0, 1, 0, 0]])
    masked_x = mask2d(path, x, background.clone())
    masked_ref = torch.tensor(
        [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]
    )
    assert torch.all(masked_x.eq(masked_ref))

    path = np.array([[0, 0, 1, 0]])
    masked_x = mask2d(path, x, background.clone())
    masked_ref = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]]
    )
    assert torch.all(masked_x.eq(masked_ref))

    path = np.array([[0, 0, 0, 1]])
    masked_x = mask2d(path, x, background.clone())
    masked_ref = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]
    )
    assert torch.all(masked_x.eq(masked_ref))


def test_mask_multiple_levels():
    x = torch.ones(1, 4, 4)
    background = torch.zeros(1, 4, 4)

    path = np.array([[1, 0, 0, 0], [0, 0, 0, 0]])
    masked_x = mask2d(path, x, background)
    assert torch.all(masked_x.eq(background))

    path = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    masked_x = mask2d(path, x, background)
    masked_ref = torch.tensor(
        [[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    )
    assert torch.all(masked_x.eq(masked_ref))

    path = np.array([[1, 0, 0, 0], [1, 0, 0, 1]])
    masked_x = mask2d(path, x, background)
    masked_ref = torch.tensor(
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    )
    assert torch.all(masked_x.eq(masked_ref))


def test_mask2str():
    str_ref = "1000"

    _str = mask2str(np.array([1, 0, 0, 0]))
    assert str_ref == _str


def test_str2mask():
    ref_mask = np.array([1, 0, 0, 0])

    mask = str2mask("1000")
    assert (mask == ref_mask).all()


def test_shapley_phi():
    logits_dictionary = {
        "10": 1,
        "00": 0,
        "11": 1,
        "01": 0,
    }
    feature_a = np.array([1, 0])
    feature_b = np.array([0, 1])
    masks = make_masks(2)

    phi_a = shapley_phi(logits_dictionary, feature_a, masks)
    phi_b = shapley_phi(logits_dictionary, feature_b, masks)
    assert phi_a == 1 and phi_b == 0

    logits_dictionary = {
        "10": 1,
        "00": 0,
        "11": 1,
        "01": 1,
    }

    phi_a = shapley_phi(logits_dictionary, feature_a, masks)
    phi_b = shapley_phi(logits_dictionary, feature_b, masks)
    assert phi_a == phi_b == 0.5


def test_child_scores():
    features = hshap_features(2)
    masks = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])

    label_logits = torch.tensor([[0], [1], [1], [0]])

    children = children_scores(label_logits, masks=masks, features=features)
    assert torch.equal(children, torch.tensor([1, 0]).float())

    label_logits = torch.tensor([[0], [1], [1], [1]])

    children = children_scores(label_logits, masks=masks, features=features)
    assert torch.equal(children, torch.tensor([0.5, 0.5]).float())
