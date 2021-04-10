import numpy as np
import torch
from torch import Tensor
from hshap.utils import (
    hshap_features,
    make_masks,
    enumerate_batches,
    children_scores,
    mask,
)
from typing import Callable
import time


class Node:
    def __init__(self, path: np.ndarray, score: int = 1) -> None:
        self.path = path
        self.score = score

    def masked_inputs(self, masks: np.ndarray, x: Tensor, background: Tensor) -> Tensor:
        """
        Mask input with all the masks required to compute Shapley values
        """
        d = len(x.shape)
        q = list(np.ones(d + 1, dtype=np.integer))
        q[0] = len(masks)

        masked_inputs = background.repeat(q)
        masked_inputs = torch.stack(
            [
                mask(
                    np.concatenate((self.path, np.expand_dims(_mask, axis=0)), axis=0),
                    x,
                    _x,
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
        M: int = 4,
    ) -> None:
        """
        Initialize explainer
        """
        self.model = model
        self.background = background
        self.size = (self.background.shape[1], self.background.shape[2])
        self.stop_l = np.log(min(self.size) / min_size) // np.log(2) + 1
        self.M = M
        self.masks = make_masks(M)
        self.features = hshap_features(M)

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
    ) -> np.ndarray:
        """
        Explain image
        """
        # Define auxilliary variables
        batch_size = 2
        # Initialize root node
        root_node = Node(np.array([[1, 1, 1, 1]]))
        leafs = []
        # Define the first level
        level = [root_node]
        L = len(level)
        while L > 0:
            layer_scores = np.zeros((L, self.M))
            for batch_id, batch in enumerate_batches(level, batch_size):
                l = len(batch)
                with torch.no_grad():
                    batch_input = torch.cat(
                        [
                            node.masked_inputs(self.masks, x, self.background)
                            for node in batch
                        ],
                        0,
                    )
                    batch_outputs = self.model(batch_input).cpu()
                    batch_outputs = batch_outputs.view(
                        (l, 2 ** self.M, batch_outputs.size(1))
                    )
                    for i, node in enumerate(batch):
                        node_outputs = batch_outputs[i]
                        node_logits = torch.nn.Softmax(dim=1)(node_outputs)
                        label_logits = node_logits[:, label]
                        layer_scores[batch_id * batch_size + i] = children_scores(
                            label_logits
                        )
                    torch.cuda.empty_cache()
                t0 = time.time()

            flat_layer_scores = layer_scores.flatten()
            if threshold_mode == "absolute":
                if any(flat_layer_scores > threshold):
                    masked_layer_scores = np.ma.masked_greater(
                        flat_layer_scores, threshold
                    ).mask.reshape(layer_scores.shape)
                else:
                    masked_layer_scores = np.zeros(layer_scores.shape)
            if threshold_mode == "relative":
                t = np.percentile(flat_layer_scores, threshold)
                if t <= 0:
                    masked_layer_scores = np.ma.masked_greater(
                        flat_layer_scores, t
                    ).mask.reshape(layer_scores.shape)
                else:
                    masked_layer_scores = np.ma.masked_greater_equal(
                        flat_layer_scores, t
                    ).mask.reshape(layer_scores.shape)

            next_level = []
            for i, node in enumerate(level):
                for j, relevant in enumerate(masked_layer_scores[i]):
                    if relevant == True:
                        child_score = layer_scores[i, j]
                        feature = self.features[j]
                        child = Node(
                            np.concatenate(
                                (node.path, np.expand_dims(feature, axis=0)), axis=0
                            ),
                            node.score * child_score,
                        )
                        if self.is_leaf(child) == True:
                            leafs.append(child)
                        else:
                            next_level.append(child)
            level = next_level
            L = len(level)

        saliency_map = torch.zeros(1, self.size[0], self.size[1])
        for leaf in leafs:
            saliency_map += mask(
                leaf.path,
                torch.ones(1, self.size[0], self.size[1]),
                torch.zeros(1, self.size[0], self.size[1]),
            )
        return saliency_map[0].numpy(), leafs
