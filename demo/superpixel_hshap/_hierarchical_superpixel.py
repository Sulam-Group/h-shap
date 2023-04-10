import torch
import math
import time
import numpy as np
from torch import Tensor

coords = lambda k, x, y: (k // y, k % x)


def find(parent, v):
    p = parent[v]
    if p != v:
        parent[v] = find(parent, p)
    return parent[v]


def hist(flat_img: Tensor, f: list[int], k: int = 20):
    h = 1 / k
    hist = torch.zeros((k,))
    c = flat_img[f]
    bins = torch.div(c, h, rounding_mode="floor").long()
    hist[bins] += 1
    return hist


def weight(
    flat_img: Tensor, feature: int, lu: int, lv: int, j: int = 4, bins: int = 20
):
    # feature contains the ids of the pixels in the superpixel
    fu, fv = feature[-2][lu], feature[-2][lv]
    if (len(feature)) < j:
        # average the color of the pixels in the superpixel
        fu, fv = torch.mean(flat_img[fu]), torch.mean(flat_img[fv])
        # distance of the mean superpixel colors
        dc = torch.abs(fu - fv)
    else:
        # compute color histogram
        (fu, _), (fv, _) = torch.histogram(flat_img[fu], bins=bins), torch.histogram(
            flat_img[fv], bins=bins
        )
        # compute chi-square distance between histograms
        dc = 1 / 2 * torch.sum((fu - fv) ** 2 / (fu + fv + 1e-6))

    return dc.item()


def img2graph(img: Tensor):
    n, m = img.size()
    V = torch.arange(n * m)
    I, J = torch.div(V, m, rounding_mode="floor"), torch.remainder(V, m)

    E = []
    dj = 1
    di = m
    for k, (i, j) in enumerate(zip(I, J)):
        if i < n - 1:
            s_neighbor = k + di
            e = (k, s_neighbor)
            E.append(e)
        if j < m - 1:
            e_neighbor = k + dj
            e = (k, e_neighbor)
            E.append(e)
    return V, E


def hierarchical_superpixel(img: Tensor, k: int = 4):
    flat_img = img.flatten()

    # 0. Convert image to graph
    V, E = img2graph(img)
    n_V = len(V)
    n_tree = n_V

    # initially, each vertex is a tree
    parent = [i for i in range(n_V)]
    feature = [[i] for i in range(n_V)]

    mst = []
    while n_tree > k:
        # 1. Find cheapest outgoing edge for each vertex
        cheapest = [(-1, -1, torch.inf) for _ in range(n_V)]
        for e in E:
            u, v = e

            pu, pv = find(parent, u), find(parent, v)
            if pu != pv:
                w = torch.abs(
                    torch.mean(flat_img[feature[pu]])
                    - torch.mean(flat_img[feature[pv]])
                ).item()
                if w < cheapest[pu][2]:
                    cheapest[pu] = (u, v, w)
                if w < cheapest[pv][2]:
                    cheapest[pv] = (v, u, w)

        # 2. Connect trees with min outgoing edges in MST
        for k in range(n_V):
            u, v, _ = cheapest[k]

            pu, pv = find(parent, u), find(parent, v)
            if pu != pv:
                mst.append((u, v))
                if pu < pv:
                    u, v = v, u
                    pu, pv = pv, pu
                parent[u] = pv
                feature[pv] += feature[pu]

                n_tree -= 1
                if n_tree == k:
                    break
    return mst, parent, feature
