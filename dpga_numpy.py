import numpy as np
from mpi4py import MPI
from scipy import linalg
import time
import sys
from small_world import small_world
import pandas as pd


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stop1 = 1e-4
stop2 = 1e-3

"dense lasso"
def dense_matrix(N, e, n):
    seed = 0
    G, E = small_world(N, e, seed)
    d_min = min(np.diag(G))
    edges = E.shape[0]
    m_i = int(n / (2 * N))
    d_i = G.diagonal()[rank]
    np.random.seed(rank)
    A_i = np.random.randn(m_i, n)
    A_i /= np.sqrt(np.sum(A_i ** 2, 0))
    np.random.seed(rank)
    x_i = np.random.randn(n, 1)
    np.random.seed(rank)
    b_i = np.dot(A_i, x_i) + 1e-2 * np.random.randn(m_i, 1)
    lam = 0.1 * np.max(np.abs(A_i.T.dot(b_i)))
    l_i = (linalg.norm(A_i, 2)) ** 2  # ||A_i||_2^2
    K = 1000  # number of iterations
    gamma_i = np.sqrt((2.6 * N) / (edges * d_min))
    c_i = (l_i + gamma_i * d_i) ** (-1)
    gamma_ii = 0
    for j in range(d_i):
        gamma_ii += ((gamma_i * gamma_i) / (gamma_i + gamma_i))
    gamma_ij = -(gamma_i * gamma_i) / (gamma_i + gamma_i)
    pi = np.zeros((n, 1), np.float64)
    return N, d_i, A_i, b_i, lam, l_i, K, x_i, gamma_ii, gamma_ij, c_i, pi, G, E


def proximal(x_i, lam, N, c_i, pi, si, A_i, b_i):
    xbar_i = x_i - c_i * (pi + si + A_i.T.dot(A_i.dot(x_i) - b_i))
    x_prox = np.multiply(np.sign(xbar_i), np.fmax(abs(xbar_i) - c_i * lam, 0))
    return x_prox

def dpga(N, e, n):
    N, d_i, A_i, b_i, lam, l_i, K, x_i, gamma_ii, gamma_ij, c_i, pi, G, E = dense_matrix(N, e, n)
    t0 = time.time()
    x_j = np.zeros((n, d_i), dtype=np.float64)
    j = 0
    for i in range(E.shape[0]):
        if E[i, rank] == 1:
            comm.send(x_i, dest=(np.where(E[i, :] == -1))[0])
            x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == -1))[0])
            j += 1
        elif E[i, rank] == -1:
            x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == 1))[0])
            comm.send(x_i, dest=(np.where(E[i, :] == 1))[0])
            j += 1
    si = np.multiply(gamma_ii, x_i)
    for i in range(d_i):
        si += np.multiply(gamma_ij, x_j[:, [i]])

    xbar_k = np.zeros((n, K+1), np.float64)
    xbar_k[:, [0]] = x_i

    for i in range(d_i):
        xbar_k[:, [0]] += x_j[:, [i]]

    xbar_k[:, [0]] = xbar_k[:, [0]]/(d_i + 1)

    for k in range(K):
        x_i = proximal(x_i, lam, N, c_i, pi, si, A_i, b_i)
        j = 0
        for i in range(E.shape[0]):
            if E[i, rank] == 1:
                comm.send(x_i, dest=(np.where(E[i, :] == -1))[0])
                x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == -1))[0])
                j += 1
            elif E[i, rank] == -1:
                x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == 1))[0])
                comm.send(x_i, dest=(np.where(E[i, :] == 1))[0])
                j += 1
        si = np.multiply(gamma_ii, x_i)
        for i in range(d_i):
            si += np.multiply(gamma_ij, x_j[:, [i]])
        pi += si
        xbar_k[:, [k + 1]] = x_i
        for i in range(d_i):
            xbar_k[:, [k+1]] += x_j[:, [i]]
        xbar_k[:, [k + 1]] = xbar_k[:, [k + 1]]/(d_i + 1)
        eps_1 = linalg.norm((x_i - xbar_k[:, [k+1]]), 2)
        for i in range(d_i):
            eps_1 += linalg.norm((x_j[:,[i]] - xbar_k[:, [k+1]]), 2)
        eps_1 = eps_1/((d_i + 1)*np.sqrt(n))
        eps_2 = linalg.norm((xbar_k[:,[k+1]]-xbar_k[:,  [k]]), 2)/np.sqrt(n)
        if eps_1 <= stop1 and eps_2 <= stop2:
            print('elapsed time', time.time() - t0)
            print('n:', n, 'iterations:', k)
            sys.stdout.flush()
            MPI.Finalize()

if __name__ == "__main__":
    _, N_str, e_str, n_str = sys.argv
    N = int(N_str)
    e = int(e_str)
    n = int(n_str)
    if rank == 0:
        print("Nodes:", N, "Add edges:", e, 'size n:', n)
    dpga(N, e, n)
