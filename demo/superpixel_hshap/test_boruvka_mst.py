from boruvka_mst import boruvka_mst

V = range(5)
E = []
E.append([0, 1, 8])
E.append([0, 2, 5])
E.append([1, 2, 9])
E.append([1, 3, 11])
E.append([2, 3, 15])
E.append([2, 4, 10])
E.append([3, 4, 7])

mst, parent = boruvka_mst(V, E)
print(mst)
print(parent)
