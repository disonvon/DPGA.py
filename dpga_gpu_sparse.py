import numpy as np
from mpi4py import MPI
from scipy import linalg
import time
import tensorflow as tf
import sys
import scipy.sparse as sp
from small_world import small_world

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stop1 = 1e-4
stop2 = 1e-3

"dense lasso"


def sparse_matrix(N, e, n):
    seed = 0
    G, E = small_world(N, e, seed)
    d_min = min(np.diag(G))
    edges = E.shape[0]
    m_i = int(n / (2 * N))
    d_i = G.diagonal()[rank]
    np.random.seed(rank)
    A_i = sp.rand(m_i, n, 0.1)
    np.random.seed(rank)
    A_i.data = np.float32(np.random.randn(A_i.nnz))
    N_i = A_i.copy()
    N_i.data = N_i.data ** 2
    A_i = A_i * sp.diags([1 / np.sqrt(np.ravel(N_i.sum(axis=0)))], [0])
    np.random.seed(rank)
    x_i = np.random.randn(n, 1)
    np.random.seed(rank)
    b_i = A_i * sp.rand(n, 1, 0.1) + 1e-2 * np.random.randn(m_i, 1)
    lam = 0.1 * np.max(np.abs(A_i.T.dot(b_i)))
    A_i_temp = sp.csr_matrix.todense(A_i)
    l_i = (linalg.norm(A_i_temp, 2)) ** 2  # ||A_i||_2^2
    K = 1000  # number of iterations
    gamma_i = np.sqrt((2.6 * N) / (edges * d_min))
    c_i = (l_i + gamma_i * d_i) ** (-1)
    gamma_ii = 0
    for j in range(d_i):
        gamma_ii += ((gamma_i * gamma_i) / (gamma_i + gamma_i))
    gamma_ij = -(gamma_i * gamma_i) / (gamma_i + gamma_i)
    pi = np.zeros((n, 1), np.float64)
    return N, d_i, A_i, b_i, lam, l_i, K, x_i, gamma_ii, gamma_ij, c_i, pi, G, E


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def graph_gpu_sparse(A_i, lam, c_i, b_i, n_i):
    tf.reset_default_graph()
    with tf.device('/gpu:0'):
        A_i_t = convert_sparse_matrix_to_sparse_tensor(A_i)
        lam_t = tf.constant(lam, dtype=tf.float32)
        b_i_t = tf.constant(b_i, dtype=tf.float32)
        c_i_t = tf.constant(c_i, dtype=tf.float32)
        zeros = tf.zeros((1, 1), dtype=tf.float32)
        p = tf.placeholder(dtype=tf.float32, shape=[n_i, 1])
        s = tf.placeholder(dtype=tf.float32, shape=[n_i, 1])
        x = tf.placeholder(dtype=tf.float32, shape=[n_i, 1])
        temp = tf.sparse_tensor_dense_matmul(A_i_t, x) - b_i_t
        xbar_i = x - c_i_t * (p + s + tf.sparse_tensor_dense_matmul(A_i_t, temp, adjoint_a=True))
        prox = tf.multiply(tf.sign(xbar_i), tf.maximum(zeros, tf.abs(xbar_i) - lam_t * c_i_t))
        return prox, p, s, x


def sess_run(sess, prox, p, s, x, pi, si, x_i):
    x_i = sess.run(prox, feed_dict={p: pi, s: si, x: x_i})
    return x_i


def dpga(N, e, n):
    N, d_i, A_i, b_i, lam, l_i, K, x_i, gamma_ii, gamma_ij, c_i, pi, G, E = sparse_matrix(N, e, n)
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

    xbar_k = np.zeros((n, K + 1), np.float64)
    xbar_k[:, [0]] = x_i
    for i in range(d_i):
        xbar_k[:, [0]] += x_j[:, [i]]
    xbar_k[:, [0]] = xbar_k[:, [0]] / (d_i + 1)  # ?????????????????

    prox_c, p_c, s_c, x_c = graph_gpu_sparse(A_i, lam, c_i, b_i, n)
    sess = tf.InteractiveSession()
    for k in range(K):
        x_i = sess_run(sess, prox_c, p_c, s_c, x_c, pi, si, x_i)
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
            xbar_k[:, [k + 1]] += x_j[:, [i]]
        xbar_k[:, [k + 1]] = xbar_k[:, [k + 1]] / (d_i + 1)
        eps_1 = linalg.norm((x_i - xbar_k[:, [k + 1]]), 2)
        for i in range(d_i):
            eps_1 += linalg.norm((x_j[:, [i]] - xbar_k[:, [k + 1]]), 2)
        eps_1 = eps_1 / ((d_i + 1) * np.sqrt(n))
        eps_2 = linalg.norm((xbar_k[:, [k + 1]] - xbar_k[:, [k]]), 2) / np.sqrt(n)

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
        print "Nodes:", N, "Add edges:", e, 'size n:', n
    dpga(N, e, n)
