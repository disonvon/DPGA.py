"""
generate small_world for DPGA
small_world: add random edges after forming a circle
"""

import numpy as np
import sys

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def small_world(n, e, seed):
    G = np.zeros((n, n), dtype=np.int)
    E = np.zeros((n + e, n), dtype=np.int)

    if n <= 3:
        raise Exception('nodes need to be > 3')

    ind = []
    ind.append(n-4)
    for i in range(n-1):
        G[i, i+1] = -1
        if i <= n - 4:
            ind.append(ind[-1]+n-2-i-1)#??????/-i or -i-1
            # ind = [ind, ind[-1] + n -2 - i]
    G[0, n-1] = -1
    temp = (n**2-3*n)/2
    np.random.seed(seed)
    loc = np.random.permutation(range(temp))
    loc = loc[0:e]
    for edge in range(e):
        i = 0
        for temp in ind:
            if temp >= loc[edge]:
                row = i
                break
            i += 1
        if row == 0:
            col = row + 2 + loc[edge]
        else:
            col = row + 2 + loc[edge] - ind[row - 1] - 1
        G[row, col] = -1
    G = G + G.T
    G += np.diag(-np.sum(G, 0),0)
    print 'G', G

    k = 0
    for i in range(n-1):
        ind_to = indices(G[i, :], lambda x: x!= 0)
        for j in ind_to:
            if j > i:
                E[k, i] = 1
                E[k, j] = -1
                k = k+1
    print 'E', E



if __name__ == "__main__":
    _, n_str, e_str, seed_str = sys.argv
    n = int(n_str)
    e = int(e_str)
    seed = int(seed_str)
    print "generage small world (random graph), \nNodes:", n, "Add edges:", e, 'Random seed:', seed
    small_world(n, e, seed)