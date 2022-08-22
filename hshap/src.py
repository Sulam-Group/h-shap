import numpy as np
import torch
from torch import Tensor
from typing import Callable, List
from .utils import (
    hshap_features,
    shapley_matrix,
    mask_input_,
    mask_features_,
    mask_map_,
)


class Explainer:
    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        background: Tensor,
    ) -> None:
        self.model = model
        self.background = background
        self.size = (
            self.background.size(0),
            self.background.size(1),
            self.background.size(2),
        )
        self.gamma = 4
        self.features = hshap_features(self.gamma)
        self.W = shapley_matrix(self.gamma, device=background.device)

    def masked_input_(
        self,
        path: np.ndarray,
        root_input: Tensor,
        root_coords: np.ndarray,
    ) -> Tensor:
        mask_input_(
            input=root_input,
            path=path,
            background=self.background,
            root_coords=root_coords,
        )

        feature_mask = torch.zeros_like(root_input, dtype=torch.bool).repeat(
            len(self.features) + 1, 1, 1, 1
        )
        mask_features_(
            feature_mask=feature_mask,
            root_coords=root_coords,
        )
        m12 = torch.logical_or(feature_mask[1], feature_mask[2]).unsqueeze_(0)
        m13 = torch.logical_or(feature_mask[1], feature_mask[3]).unsqueeze_(0)
        m14 = torch.logical_or(feature_mask[1], feature_mask[4]).unsqueeze_(0)
        m23 = torch.logical_or(feature_mask[2], feature_mask[3]).unsqueeze_(0)
        m24 = torch.logical_or(feature_mask[2], feature_mask[4]).unsqueeze_(0)
        m34 = torch.logical_or(feature_mask[3], feature_mask[4]).unsqueeze_(0)
        m123 = torch.logical_or(m12, m13)
        m124 = torch.logical_or(m12, m14)
        m134 = torch.logical_or(m13, m14)
        m234 = torch.logical_or(m23, m24)
        m1234 = torch.logical_or(m123, m124)
        m = torch.cat(
            [
                feature_mask[:5],
                m12,
                m13,
                m14,
                m23,
                m24,
                m34,
                m123,
                m124,
                m134,
                m234,
                m1234,
            ]
        )
        return torch.where(m, root_input, self.background)

    def explain(
        self,
        x: Tensor,
        label: int,
        s: int,
        threshold_mode: str = "absolute",
        threshold: float = 0.0,
        softmax_activation: bool = True,
        batch_size: int = 2,
        binary_map: bool = False,
        roll_row: int = 0,
        roll_column: int = 0,
        **kwargs,
    ) -> Tensor:
        nodes = np.ones((1, 1, 4), dtype=np.bool_)
        scores = torch.ones((1,), device=self.background.device).float()
        root_coords = np.array(
            [[[0, 0], [self.size[1], self.size[2]]]], dtype=np.uint16
        )

        if roll_row != 0 or roll_column != 0:
            x = torch.roll(x, shifts=(-roll_row, -roll_column), dims=(1, 2))
        root_inputs = x.unsqueeze_(0)

        stop_l = int(np.log2(min(self.size[1], self.size[2]) / s)) + 2
        while nodes.shape[1] < stop_l:
            scores = scores.unsqueeze_(1).repeat((1, self.gamma))

            for batch_start_id in range(0, len(nodes), batch_size):
                batch = nodes[batch_start_id : batch_start_id + batch_size]
                batch_input = torch.empty_like(self.background).repeat(
                    len(batch), 2**self.gamma, 1, 1, 1
                )

                for n, node in enumerate(batch):
                    batch_input[n] = self.masked_input_(
                        node[-1],
                        root_inputs[batch_start_id + n],
                        root_coords[batch_start_id + n],
                    )

                    if roll_row != 0 or roll_column != 0:
                        batch_input[n] = torch.roll(
                            batch_input[n], shifts=(roll_row, roll_column), dims=(2, 3)
                        )

                F = self.model(
                    batch_input.view(
                        len(batch) * 2**self.gamma,
                        self.size[0],
                        self.size[1],
                        self.size[2],
                    ),
                    **kwargs,
                )

                if softmax_activation:
                    F = torch.nn.functional.softmax(F, dim=1)

                F = F[:, label]
                F = F.view((-1, 1, 2**self.gamma))
                scores[batch_start_id : batch_start_id + batch_size].mul_(
                    torch.matmul(F, self.W).squeeze_(1)
                )

            if threshold_mode == "absolute":
                masked_scores = scores > threshold
            if threshold_mode == "relative":
                t = torch.quantile(scores, threshold / 100, dim=None)
                if t <= 0:
                    masked_scores = scores > t
                else:
                    masked_scores = scores >= t

            i, j = masked_scores.nonzero(as_tuple=True)
            del masked_scores

            _i, _j = i.size(0), j.size(0)
            ic, jc = i.cpu(), j.cpu()

            if _i == 0 and _j == 0:
                raise ValueError("Could not find any important nodes.")
            if _i == 1 and _j == 1:
                nodes = np.concatenate(
                    (nodes[None, ic], self.features[None, jc]), axis=1, dtype=np.bool_
                )
                root_coords = root_coords[None, ic]
            else:
                nodes = np.concatenate(
                    (nodes[ic], self.features[jc]), axis=1, dtype=np.bool_
                )
                root_coords = root_coords[ic]

            scores = scores[i, j]
            root_inputs = root_inputs[i]

        saliency_map = torch.zeros(1, self.size[1], self.size[2])
        _scores = scores.tolist()

        for n, s, c in zip(nodes, _scores, root_coords):
            mask_map_(
                map=saliency_map,
                path=n[-1],
                score=s if not binary_map else 1,
                root_coords=c,
            )

        if roll_row != 0 or roll_column != 0:
            saliency_map = torch.roll(
                saliency_map, shifts=(roll_row, roll_column), dims=(1, 2)
            )
        return saliency_map

    def cycle_explain(
        self,
        x: Tensor,
        label: int,
        s: int,
        R: List[int],
        A: List[int],
        **kwargs,
    ):
        saliency_map = torch.zeros(1, self.size[1], self.size[2])
        for r in R:
            for a in A:
                if r == 0 and a != 0:
                    continue
                else:
                    roll_row = -int(r * np.sin(a))
                    roll_column = int(r * np.cos(a))

                    saliency_map += self.explain(
                        x.clone(),
                        label=label,
                        s=s,
                        roll_row=roll_row,
                        roll_column=roll_column,
                        **kwargs,
                    )
        saliency_map /= ((len(R) - 1) * len(A)) + 1
        saliency_map
        return saliency_map
