from typing import Generator, Iterable, Tuple
import torch
import numpy as np
from torch import Tensor
from itertools import permutations
from functools import reduce

factorial = np.math.factorial


def enumerate_batches(
    collection: Iterable, batch_size: int
) -> Generator[Tuple[int, list], None, None]:
    """
    Batch enumerator
    """
    L = len(collection)
    for i, first_el in enumerate(range(0, L, batch_size)):
        last_el = first_el + batch_size
        if last_el < L:
            yield i, collection[first_el:last_el]
        else:
            yield i, collection[first_el:]


def hshap_features(gamma: int) -> Tensor:
    """
    Make the required features
    """
    return torch.eye(gamma).long()


def make_masks(gamma: int) -> Tensor:
    """
    Make all required masks to compute Shapley values given the number of features gamma
    and order them by their rank, where the rank is the integer obtain by concatenating
    the indices of the nonzero elments in the mask
    """
    masks = []
    for i in range(1, gamma + 1):
        masks.extend(list(set(permutations((gamma - i) * [0] + i * [1]))))
    masks = torch.tensor(masks).long()
    rank = torch.tensor(
        [int("".join(map(str, (m.nonzero() + 1).squeeze(1).tolist()))) for m in masks]
    )
    masks = masks[rank.argsort()]
    masks = torch.cat((torch.zeros((1, gamma)).long(), masks))
    return masks


def w(c: int, gamma: int) -> int:
    """
    Compute the weight of a subset of features of cardinality c
    """
    return factorial(c) * factorial(gamma - c - 1) / factorial(gamma)


def shapley_matrix(gamma: int) -> Tensor:
    """
    Compose the matrix to compute the Shapley values.
    This function assumes that masks are ordered as per `make_masks`
    definition of rank
    """
    if gamma != 4:
        raise NotImplementedError
    # construct matrix as copies of first row
    W = torch.tensor(
        [
            [-w(0, gamma), w(0, gamma)]
            + 3 * [-w(1, gamma)]
            + 3 * [w(1, gamma)]
            + 3 * [-w(2, gamma)]
            + 3 * [w(2, gamma)]
            + [-w(3, gamma), w(3, gamma)]
        ]
    ).repeat(gamma, 1)
    # update second row
    W[1, 1] = -w(1, gamma)
    W[1, 2] = w(0, gamma)
    W[1, 6] = -w(2, gamma)
    W[1, 7] = -w(2, gamma)
    W[1, 8] = w(1, gamma)
    W[1, 9] = w(1, gamma)
    W[1, -3] = -w(3, gamma)
    W[1, -2] = w(2, gamma)
    # update second row
    W[2, 1] = -w(1, gamma)
    W[2, 3] = w(0, gamma)
    W[2, 5] = -w(2, gamma)
    W[2, 7] = -w(2, gamma)
    W[2, 8] = w(1, gamma)
    W[2, 10] = w(1, gamma)
    W[2, -4] = -w(3, gamma)
    W[2, -2] = w(2, gamma)
    # update third row
    W[3, 1] = -w(1, gamma)
    W[3, 4] = w(0, gamma)
    W[3, 5] = -w(2, gamma)
    W[3, 6] = -w(2, gamma)
    W[3, 9] = w(1, gamma)
    W[3, 10] = w(1, gamma)
    W[3, -5] = -w(3, gamma)
    W[3, -2] = w(2, gamma)

    return W.transpose(0, 1)


def mask2d(
    path: Tensor, x: Tensor, _x: Tensor, r: float = 0, alpha: float = 0
) -> Tensor:
    """
    Creates a masked copy of x based on node.path and the specified background
    """

    if len(path) == 0:
        return x

    if path[-1].sum() == 0:
        return _x
    else:
        coords = np.array([[0, 0], [_x.size(1), _x.size(2)]], dtype=int)
        for level in path[:-1]:
            if level.sum() == 1:
                center = (
                    (coords[0][0] + coords[1][0]) / 2,
                    (coords[0][1] + coords[1][1]) / 2,
                )
                feature_id = level.nonzero().squeeze()
                (feature_row, feature_column) = (feature_id // 2, feature_id % 2)
                coords[0][0] = center[0] if feature_row == 1 else coords[0][0]
                coords[0][1] = center[1] if feature_column == 1 else coords[0][1]
                coords[1][0] = center[0] if (1 - feature_row) == 1 else coords[1][0]
                coords[1][1] = center[1] if (1 - feature_column) == 1 else coords[1][1]
        level = path[-1]
        center = ((coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2)
        feature_ids = level.nonzero().squeeze(1)
        feature_mask = torch.zeros_like(x)
        for feature_id in feature_ids:
            (feature_row, feature_column) = (feature_id // 2, feature_id % 2)
            feature_coords = coords.copy()
            feature_coords[0][0] = (
                center[0] if feature_row == 1 else feature_coords[0][0]
            )
            feature_coords[0][1] = (
                center[1] if feature_column == 1 else feature_coords[0][1]
            )
            feature_coords[1][0] = (
                center[0] if (1 - feature_row) == 1 else feature_coords[1][0]
            )
            feature_coords[1][1] = (
                center[1] if (1 - feature_column) == 1 else feature_coords[1][1]
            )
            # feature_mask
            feature_mask[
                :,
                feature_coords[0][0] : feature_coords[1][0],
                feature_coords[0][1] : feature_coords[1][1],
            ] = 1
        # roll the feature mask if desired
        if r != 0 or alpha != 0:
            column_offset = int(r * np.cos(alpha))
            row_offset = -int(r * np.sin(alpha))
            feature_mask = torch.roll(feature_mask, row_offset, dims=1)
            feature_mask = torch.roll(feature_mask, column_offset, dims=2)
        _x = feature_mask * x + (1 - feature_mask) * _x
        return _x
