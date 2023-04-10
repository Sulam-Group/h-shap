import torch
from torch import Tensor
from boruvka_mst import UnionFind


class SH(object):
    def __init__(self, img):
        self.img = img
        self.flat_img = self.img.view(-1)
        self.V, self.E = self.__img2graph__(img)

    def __img2graph__(self, img: Tensor):
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

    def __weight__(self, pu, pv):
        fu = torch.mean(self.flat_img[self.uf.parent == pu])
        fv = torch.mean(self.flat_img[self.uf.parent == pv])
        dc = torch.abs(fu - fv)
        return dc

    def build(self, k: int):
        self.uf = UnionFind(self.V)
        mst = []
        while self.uf.count > k:
            # 1. Find cheapest outgoing edge for each component
            cheapest = [(-1, -1, torch.inf) for _ in range(len(self.V))]
            for e in self.E:
                u, v = e
                pu, pv = self.uf.find(u), self.uf.find(v)

                if pu != pv:
                    w = self.__weight__(pu, pv)
                    if w < cheapest[pu][2]:
                        cheapest[pu] = (u, v, w)
                    if w < cheapest[pv][2]:
                        cheapest[pv] = (u, v, w)

            # 2. Connect components with min outgoing edges in MST
            for e in cheapest:
                u, v, _ = e
                pu, pv = self.uf.find(u), self.uf.find(v)
                if pu != pv:
                    mst.append((u, v))
                    self.uf.union(u, v)
                    if self.uf.count == k:
                        break
        return mst, self.uf.parent
