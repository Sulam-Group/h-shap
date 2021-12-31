import numpy as np
import torch
from torch import Tensor
from hshap.utils import (
    hshap_features,
    make_masks,
    shapley_matrix,
    enumerate_batches,
    mask2d,
)
from typing import Callable


class Node:
    def __init__(self, path: Tensor, score: float = 1) -> None:
        self.path = path
        self.score = score

    def masked_inputs(
        self,
        masks: Tensor,
        x: Tensor,
        background: Tensor,
        r: float = 0,
        alpha: float = 0,
    ) -> Tensor:
        """
        Mask input with all the masks required to compute Shapley values
        """
        q = list(np.ones(x.ndim + 1, dtype=np.int32))
        q[0] = len(masks)

        masked_inputs = background.repeat(q)
        masked_inputs = torch.stack(
            [
                mask2d(
                    torch.cat((self.path, _mask.unsqueeze(0)), dim=0),
                    x,
                    _x,
                    r,
                    alpha,
                )
                for _mask, _x in zip(masks, masked_inputs)
            ],
            0,
        )
        return masked_inputs


class Explainer:
    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        background: Tensor,
        min_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize explainer
        """
        self.model = model
        self.background = background
        self.size = (self.background.shape[1], self.background.shape[2])
        self.stop_l = np.log(min(self.size) / min_size) // np.log(2) + 2
        self.gamma = 4
        self.masks = make_masks(self.gamma)
        self.features = hshap_features(self.gamma)
        self.W = shapley_matrix(self.gamma)
        self.W = self.W.to(device)

    def is_leaf(self, node: Node) -> bool:
        """
        Check if node is a leaf
        """
        if len(node.path) == self.stop_l:
            return True
        else:
            return False

    def explain(
        self,
        x: Tensor,
        label: int,
        threshold_mode: str = "absolute",
        threshold: float = 0.0,
        r: float = 0.0,
        alpha: float = 0.0,
        softmax_activation: bool = True,
        binary_map: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Explain image
        """
        batch_size = 32
        root_node = Node(path=torch.ones((1, self.gamma)).long(), score=1.0)
        leafs = []
        level = [root_node]
        L = len(level)
        while L > 0:
            layer_scores = torch.empty((L, self.gamma))
            for batch_id, batch in enumerate_batches(level, batch_size):
                l = len(batch)
                batch_input = torch.cat(
                    [
                        node.masked_inputs(self.masks, x, self.background, r, alpha)
                        for node in batch
                    ],
                    0,
                )
                batch_outputs = self.model(batch_input, **kwargs)
                if softmax_activation:
                    batch_outputs = torch.nn.functional.softmax(batch_outputs, dim=1)
                label_outputs = batch_outputs[:, label]
                label_outputs = label_outputs.view((l, 1, 2 ** self.gamma))
                layer_scores[
                    batch_id * batch_size : (batch_id + 1) * batch_size
                ] = torch.matmul(label_outputs, self.W).squeeze(1)

            if threshold_mode == "absolute":
                masked_layer_scores = layer_scores > threshold
            if threshold_mode == "relative":
                t = np.percentile(layer_scores.flatten(), threshold)
                if t <= 0:
                    masked_layer_scores = layer_scores > t
                else:
                    masked_layer_scores = layer_scores >= t

            next_level = []
            for i, node in enumerate(level):
                for j, relevant in enumerate(masked_layer_scores[i]):
                    if relevant == True:
                        child_score = layer_scores[i, j]
                        feature = self.features[j]
                        child = Node(
                            torch.cat((node.path, feature.unsqueeze(0)), dim=0),
                            score=node.score * child_score,
                        )
                        if self.is_leaf(child) == True:
                            leafs.append(child)
                        else:
                            next_level.append(child)
            level = next_level
            L = len(level)

        saliency_map = torch.zeros(1, self.size[0], self.size[1])
        for leaf in leafs:
            saliency_map += mask2d(
                path=leaf.path,
                x=torch.ones(1, self.size[0], self.size[1])
                * (1 if binary_map else leaf.score),
                _x=torch.zeros(1, self.size[0], self.size[1]),
                r=r,
                alpha=alpha,
            )
        return saliency_map.squeeze(), leafs
