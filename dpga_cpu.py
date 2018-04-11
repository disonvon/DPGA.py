# -*- coding: utf-8 -*-
import numpy as np
from mpi4py import MPI
from scipy import linalg
import time
import tensorflow as tf
import sys
from small_world import small_world
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stop1 = 1e-4
stop2 = 1e-3
K = 2000

"dense lasso"
def dense_matrix(N, n):
    #generate lasso problem
    m_i = int(n / (2 * N))
    np.random.seed(rank)
    A_i = np.random.randn(m_i, n)
    A_i /= np.sqrt(np.sum(A_i ** 2, 0))
    np.random.seed(rank)
    xi = np.random.randn(n, 1)
    b_i = np.dot(A_i, xi) + 1e-2 * np.random.randn(m_i, 1)
    lam = 0.1 * np.max(np.abs(A_i.T.dot(b_i)))
    return A_i, b_i, lam, xi

def get_par(N, e, n, A_i):
    seed = 0
    # get the small world network in G and E matrix
    G, E = small_world(N, e, seed)
    # get minimum degree of G
    d_min = min(np.diag(G))
    # get the total number of edges in G
    edges = E.shape[0]
    m_i = int(n / (2 * N))
    # degree of node i in G
    d_i = G.diagonal()[rank]
    #lipshitz constant
    l_i = (linalg.norm(A_i, 2)) ** 2  # ||A_i||_2^2
    #node specific parameter for DPGA
    gamma_i = np.sqrt((2.6 * N) / (edges * d_min))
    #step size
    c_i = (l_i + gamma_i * d_i) ** (-1)
    gamma_ii = 0
    for j in range(d_i):
        gamma_ii += ((gamma_i * gamma_i) / (gamma_i + gamma_i))
    gamma_ij = -(gamma_i * gamma_i) / (gamma_i + gamma_i)
    pi = np.zeros((n, 1), np.float64)
    return G, E, d_i, l_i, gamma_i, c_i, gamma_ii, gamma_ij, pi


def graph_cpu(A_i, lam, c_i, b_i, n_i):
    #construct computation graph for proximal gradient operator on node i
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    with tf.device('/cpu:0'):
        A_i = tf.constant(A_i, dtype=tf.float64, name='A_i')
        lam = tf.constant(lam, dtype=tf.float64, name='lambda')
        b_i = tf.constant(b_i, dtype=tf.float64, name='b_i')
        c_i = tf.constant(c_i, dtype=tf.float64, name='c_i')
        zeros = tf.zeros((1, 1), dtype=tf.float64, name='zeros')
        p_i = tf.placeholder(dtype=tf.float64, shape=[n_i, 1], name='p_i')
        s_i = tf.placeholder(dtype=tf.float64, shape=[n_i, 1], name='s_i')
        x_i = tf.placeholder(dtype=tf.float64, shape=[n_i, 1], name='x_i')
        temp = tf.matmul(A_i, x_i) - b_i
        xbar_i = x_i - c_i * (p_i + s_i + tf.matmul(A_i, temp, transpose_a=True))
        prox_i = tf.multiply(tf.sign(xbar_i), tf.maximum(zeros, tf.abs(xbar_i) - lam * c_i))
        return prox_i, p_i, s_i, x_i

def sess_run(sess, prox_i, p_i, s_i, x_i, pi, si, xi):
    xi = sess.run(prox_i, feed_dict={p_i: pi, s_i: si, x_i: xi})
    return xi

def dpga_cpu(N, e, n):
    A_i, b_i, lam, xi = dense_matrix(N, n)
    G, E, d_i, l_i, gamma_i, c_i, gamma_ii, gamma_ij, pi = get_par(N, e, n, A_i)
    t0 = time.time()
    x_j = np.zeros((n, d_i), dtype=np.float64)
    j = 0
    #communicate information with their neighbors
    for i in range(E.shape[0]):
        if E[i, rank] == 1:
            comm.send(xi, dest=(np.where(E[i, :] == -1))[0])
            x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == -1))[0])
            j += 1
        elif E[i, rank] == -1:
            x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == 1))[0])
            comm.send(xi, dest=(np.where(E[i, :] == 1))[0])
            j += 1
    #get si
    si = np.multiply(gamma_ii, xi)
    for i in range(d_i):
        si += np.multiply(gamma_ij, x_j[:, [i]])

    xbar_k = np.zeros((n, K+1), np.float64)
    xbar_k[:, [0]] = xi
    prox_c, p_c, s_c, x_c = graph_cpu(A_i, lam, c_i, b_i, n)
    sess = tf.InteractiveSession()
    for i in range(d_i):
        xbar_k[:, [0]] += x_j[:, [i]]

    xbar_k[:, [0]] = xbar_k[:, [0]]/(d_i + 1)
    #DPGA iterations
    for k in range(K):
        xi = sess_run(sess, prox_c, p_c, s_c, x_c, pi, si, xi)
        j = 0
        # communicate information with their neighbors
        for i in range(E.shape[0]):
            if E[i, rank] == 1:
                comm.send(xi, dest=(np.where(E[i, :] == -1))[0])
                x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == -1))[0])
                j += 1
            elif E[i, rank] == -1:
                x_j[:, [j]] = comm.recv(source=(np.where(E[i, :] == 1))[0])
                comm.send(xi, dest=(np.where(E[i, :] == 1))[0])
                j += 1
        #update si
        si = np.multiply(gamma_ii, xi)
        for i in range(d_i):
            si += np.multiply(gamma_ij, x_j[:, [i]])
        #update pi
        pi += si
        xbar_k[:, [k + 1]] = xi
        for i in range(d_i):
            xbar_k[:, [k+1]] += x_j[:, [i]]
        xbar_k[:, [k + 1]] = xbar_k[:, [k + 1]]/(d_i + 1)
        eps_1 = linalg.norm((xi - xbar_k[:, [k+1]]), 2)
        for i in range(d_i):
            eps_1 += linalg.norm((x_j[:, [i]] - xbar_k[:, [k+1]]), 2)
        eps_1 = eps_1/((d_i + 1)*np.sqrt(n))
        eps_2 = linalg.norm((xbar_k[:, [k+1]]-xbar_k[:, [k]]), 2)/np.sqrt(n)
        #check stop condition
        if eps_1 <= stop1 and eps_2 <= stop2:
            print('elapsed time', time.time() - t0)
            print('rank', rank, 'n:', n, 'iterations:', k)
            sys.stdout.flush()
            # if rank == 0:
            #     x_dpga = pd.DataFrame(x_i, columns=list('A'))
            #     x_dpga.to_csv('x_dpga.csv')
            MPI.Finalize()

if __name__ == "__main__":
    _, N_str, e_str, n_str = sys.argv
    N = int(N_str)
    e = int(e_str)
    n = int(n_str)
    if rank == 0:
        print("Nodes:", N, "Add edges:", e, 'problem size n:', n)
    dpga_cpu(N, e, n)
