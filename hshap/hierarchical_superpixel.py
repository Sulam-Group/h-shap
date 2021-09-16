import math
import time
import numpy as np

coords = lambda k, x, y: (k // y, k % x)


def find(parent, merge, v):
    p = parent[v][merge]
    if p != v:
        parent[v][merge] = find(parent, merge, p)
    return p


def hist(flat_img, r, k=20):
    h = 1 / k
    hist = np.zeros((k,))
    c = flat_img[r]
    bins = np.floor_divide(c, h).astype(int)
    hist[bins] += 1
    return hist


def weight(flat_img, feature, merge, lu, lv, j=4):
    # feature contains the ids of the pixel in the superpixel
    fu, fv = feature[merge][lu], feature[merge][lv]
    if (merge + 1) < j:
        # average the color of the pixels in the superpixel
        mean_c = lambda r: np.mean(flat_img[r])
        fu, fv = mean_c(fu), mean_c(fv)
        # distance of the mean superpixel colors
        dc = abs(fu - fv)
    else:
        # compute color histogram
        fu, fv = hist(flat_img, fu), hist(flat_img, fv)
        # compute chi-square distance between histograms
        dc = 1 / 2 * sum([(ui - vi) ** 2 / (ui + vi + 1e-6) for ui, vi in zip(fu, fv)])

    return dc


def img2graph(img):
    x, y = img.shape

    graph = []

    V = x * y

    di = x
    dj = 1
    for k in range(V):
        i, j = coords(k, x, y)
        if i < y - 1:
            s_neighbor = k + di
            e = (k, s_neighbor)
            graph.append(e)
        if j < x - 1:
            e_neighbor = k + dj
            e = (k, e_neighbor)
            graph.append(e)

    return graph, V


def hierarchical_superpixel(img):

    T0 = time.time()
    # 1. Convert image to graph
    t0 = time.time()
    graph, V = img2graph(img)
    flat_img = img.flatten()
    n_tree = V
    # initially, each vertex is a tree
    merge = 0
    parent = [[i] for i in range(V)]
    label = [[i] for i in range(V)]
    feature = [{i: [i] for i in range(V)}]
    feature_hierarchy = [{i: [i] for i in range(V)}]
    tree_root = [i for i in range(V)]
    mst = []

    print(
        f"[TIMER] Step 1 - converting image to graph took {1e3*(time.time() - t0):.4f} ms"
    )

    # repeat until merging is complete
    while n_tree > 1:

        merge += 1
        print(f"Started MERGE #{merge}")
        parent = [p + p[-1:] for p in parent]
        label = [l + l[-1:] for l in label]
        feature.append({})
        feature_hierarchy.append({})

        # 2. Find cheapest outgoing edge for each tree
        t0 = time.time()
        cheapest = [(-1, -1, math.inf) for _ in range(n_tree)]
        for e in graph:
            u, v = e
            # print("Vertices")
            # print(u, v)
            lu, lv = label[u][merge - 1], label[v][merge - 1]
            # print("Labels")
            # print(lu, lv)
            w = weight(flat_img, feature, merge - 1, lu, lv)
            if w < cheapest[lu][2]:
                cheapest[lu] = (u, v, w)
            if w < cheapest[lv][2]:
                cheapest[lv] = (u, v, w)
            # break
        # raise NotImplementedError
        print(
            f"[TIMER] Step 2 - finding cheapest outgoing edge for each tree took {1e3*(time.time() - t0):.4f} ms"
        )

        # 3. Connect trees with min outgoing edges in MST
        t0 = time.time()
        cheapest.sort(key=lambda x: x[2])
        for t in range(n_tree):
            u, v, _ = cheapest[t]
            pu, pv = find(parent, merge, u), find(parent, merge, v)
            if pu != pv:
                mst.append((u, v))
                if pu < pv:
                    u, v = v, u
                    pu, pv = pv, pu
                parent[u][merge] = pv
        # print(parent)
        print(
            f"[TIMER] Step 3 - connecting trees with min outgoing edges in MST took {1e3*(time.time() - t0):.4f} ms"
        )

        # 4. Update trees
        t0 = time.time()
        new_n_tree = 0
        for t in range(n_tree):
            v = tree_root[t]  # old root
            root = find(parent, merge, v)  # new root

            if v == root:
                label[root][merge] = new_n_tree
                tree_root[new_n_tree] = v
                new_n_tree += 1
            else:
                label[v][merge] = label[root][merge]

            # merge features
            if label[root][merge] not in feature[merge]:
                feature[merge][label[root][merge]] = []
            feature[merge][label[root][merge]] += feature[merge - 1][
                label[v][merge - 1]
            ]
            # extend feature hierarchy
            # if label[root][merge] not in feature_hierarchy[merge]:
            #     feature_hierarchy[merge][label[root][merge]] = []
            # feature_hierarchy[merge][label[root][merge]].append(label[v][merge - 1])
        n_tree = new_n_tree
        # print(label)
        # print(feature[merge])
        print(f"[TIMER] Step 4 - updating trees took {1e3*(time.time() - t0):.4f} ms")

        # 5. Update graph
        t0 = time.time()
        new_graph = set()
        for e in graph:
            u, v = e
            pu, pv = find(parent, merge, u), find(parent, merge, v)

            if pu == pv:
                continue
            else:
                if pu < pv:
                    u, v = v, u
                    pu, pv = pv, pu
                e = (pu, pv)
                if e not in new_graph:
                    new_graph.add(e)
        new_graph = list(new_graph)
        # print(len(new_graph))
        print(f"[TIMER] Step 5 - updating edges took {1e3*(time.time() - t0):.4f} ms")

        print(f"There are {n_tree} trees left")

        graph = new_graph

    print(f"Computed superpixel hierarchy in {1e3*(time.time()-T0):.4f} ms")
    return graph, mst, parent, feature, feature_hierarchy
