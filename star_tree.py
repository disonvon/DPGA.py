import numpy as np
import sys


def tree(N):
    E = np.zeros((N-1, N), dtype=np.int)
    G = np.zeros((N, N), dtype=np.int)
    for i in range(N - 1):
        E[i, 0] = 1
        E[i, i + 1] = -1
        if i == 0:
            G[0, 0] = N - 1
            G[0, 1::] = -1
        else:
            G[i, i] = 1
            G[i, 0] = -1
    G[N - 1, N - 1] = 1
    G[N - 1, 0] = -1
    print 'edges', E, '\nGraph', G
    return E, G


if __name__ == '__main__':
    _, N_str = sys.argv
    N = int(N_str)
    print("generate a tree network, \nNodes:", N)
    tree(N)


