"""
solve centre Lasso Problem
for DPGA
lam_i = 0.1||A*b||_inf
"""

import numpy as np
import cvxpy as cvx
import sys
import time
import pandas as pd
from scipy import linalg



def lasso_dense(n, N):
    n_i = n
    m_i = int(n_i / (2 * N))
    obj = 0
    x = cvx.Variable(n)
    for i in range(N):
        np.random.seed(i)
        A_i = np.random.randn(m_i, n_i)
        A_i /= np.sqrt(np.sum(A_i ** 2, 0))
        np.random.seed(i)
        x_i = np.random.randn(n_i, 1)
        np.random.seed(i)
        b_i = np.dot(A_i, x_i) + 1e-2 * np.random.randn(m_i, 1)
        lam = 0.1 * np.max(np.abs(A_i.T.dot(b_i)))
        obj += 0.5*cvx.sum_squares(A_i*x - b_i) + lam*cvx.norm1(x)
    return x, obj


def run_cvx(x, obj):
    prob = cvx.Problem(cvx.Minimize(obj))
    t0 = time.time()
    prob.solve()
    print "solve_time: %.2f secs" % (time.time() - t0)
    print 'obj', obj.value
    # x_cvx = x.value
    # x_cvx = pd.DataFrame(x_cvx, columns=list('A'))
    # x_cvx.to_csv('x_cvx.csv')



if __name__ == "__main__":
    _, n_str, N_str = sys.argv
    n = int(n_str)
    N = int(N_str)
    print "running cvx to solve central problem, \nn=", n, 'N=', N
    run_cvx(*lasso_dense(n, N))





