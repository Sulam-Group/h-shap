import math
import torch


class UnionFind(object):
    def __init__(self, V):
        self.V = V
        self.count = len(V)
        self.parent = torch.arange(self.count)
        self.rank = torch.zeros(self.count)

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            if self.rank[pu] > self.rank[pv]:
                self.parent[pv] = pu
            elif self.rank[pu] < self.rank[pv]:
                self.parent[pu] = pv
            else:
                self.parent[pv] = pu
                self.rank[pu] += 1
            self.count -= 1


# implement boruvka's algorithm for minimum spanning tree
def boruvka_mst(V, E):
    uf = UnionFind(V)

    mst = []
    n_tree = len(V)
    while n_tree > 1:
        # 1. Find cheapest outgoing edge for each vertex
        cheapest = [(-1, -1, torch.inf) for _ in range(len(V))]
        for e in E:
            u, v, w = e
            pu, pv = uf.find(u), uf.find(v)

            if pu != pv:
                if w < cheapest[pu][2]:
                    cheapest[pu] = (u, v, w)
                if w < cheapest[pv][2]:
                    cheapest[pv] = (v, u, w)

        # 2. Connect trees with min outgoing edges in MST
        for e in cheapest:
            u, v, w = e
            pu, pv = uf.find(u), uf.find(v)
            if pu != pv:
                mst.append((u, v, w))
                uf.union(u, v)
                n_tree -= 1
    return mst, uf.parent
