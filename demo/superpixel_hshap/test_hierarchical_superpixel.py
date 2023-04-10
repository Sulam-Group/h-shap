import torch
from demo.superpixel_hshap._hierarchical_superpixel import img2graph


def test_img2graph():
    n, m = 3, 5

    V, E = img2graph(torch.rand(n, m))

    assert torch.eq(torch.arange(n * m), V).all()

    I, J = torch.div(V, m, rounding_mode="floor"), torch.remainder(V, m)
    assert torch.eq(
        torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 1],
                [2, 0, 2],
                [3, 0, 3],
                [4, 0, 4],
                [5, 1, 0],
                [6, 1, 1],
                [7, 1, 2],
                [8, 1, 3],
                [9, 1, 4],
                [10, 2, 0],
                [11, 2, 1],
                [12, 2, 2],
                [13, 2, 3],
                [14, 2, 4],
            ]
        ),
        torch.stack((V, I, J), dim=1),
    ).all()

    assert set(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (5, 10),
            (6, 11),
            (7, 12),
            (8, 13),
            (9, 14),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
        ]
    ) == set(E)
