import torch
import numpy as np
from torch import Tensor

factorial = np.math.factorial


def hshap_features(gamma: int) -> np.ndarray:
    return np.expand_dims(np.eye(gamma, dtype=np.bool_), axis=1)


def w(c: int, gamma: int) -> int:
    return factorial(c) * factorial(gamma - c - 1) / factorial(gamma)


def shapley_matrix(gamma: int, device: torch.device) -> Tensor:
    if gamma != 4:
        raise NotImplementedError("Only implemented for gamma = 4")
    # construct matrix as copies of first row
    W = torch.tensor(
        [
            [-w(0, gamma), w(0, gamma)]
            + 3 * [-w(1, gamma)]
            + 3 * [w(1, gamma)]
            + 3 * [-w(2, gamma)]
            + 3 * [w(2, gamma)]
            + [-w(3, gamma), w(3, gamma)]
        ],
        device=device,
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


def mask_features_(
    feature_mask: Tensor,
    root_coords: np.ndarray,
) -> None:
    center = np.mean(root_coords, axis=0, dtype=np.uint16)

    feature_mask[
        1, :, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]
    ].fill_(True)
    feature_mask[
        2, :, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]
    ].fill_(True)
    feature_mask[
        3, :, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]
    ].fill_(True)
    feature_mask[
        4, :, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]
    ].fill_(True)


def mask_input_(
    input: Tensor,
    path: np.ndarray,
    background: Tensor,
    root_coords: np.ndarray,
) -> None:
    if not np.all(path):
        center = np.mean(root_coords, axis=0, dtype=np.uint16)

        if not path[0]:
            input[
                :, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]
            ] = background[
                :, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]
            ]
        if not path[1]:
            input[
                :, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]
            ] = background[
                :, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]
            ]
        if not path[2]:
            input[
                :, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]
            ] = background[
                :, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]
            ]
        if not path[3]:
            input[
                :, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]
            ] = background[
                :, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]
            ]

        feature_id = np.nonzero(path)[0][0]
        feature_row, feature_column = feature_id // 2, feature_id % 2
        root_coords[0, 0] = center[0] if feature_row == 1 else root_coords[0, 0]
        root_coords[0, 1] = center[1] if feature_column == 1 else root_coords[0, 1]
        root_coords[1, 0] = center[0] if (1 - feature_row) == 1 else root_coords[1, 0]
        root_coords[1, 1] = (
            center[1] if (1 - feature_column) == 1 else root_coords[1, 1]
        )


def mask_map_(
    map: Tensor,
    path: np.ndarray,
    score: float,
    root_coords: np.ndarray,
):
    center = np.mean(root_coords, axis=0, dtype=np.uint16)

    if path[0]:
        map[:, root_coords[0, 0] : center[0], root_coords[0, 1] : center[1]].fill_(
            score
        )
    elif path[1]:
        map[:, root_coords[0, 0] : center[0], center[1] : root_coords[1, 1]].fill_(
            score
        )
    elif path[2]:
        map[:, center[0] : root_coords[1, 0], root_coords[0, 1] : center[1]].fill_(
            score
        )
    elif path[3]:
        map[:, center[0] : root_coords[1, 0], center[1] : root_coords[1, 1]].fill_(
            score
        )
