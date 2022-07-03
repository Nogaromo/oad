import numpy as np
from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'grandnorm', 'x'))


def gauss_newton(z, f, j, x0, k=1, tol=1e-4, max_iter=1000):
    x = np.asarray(x0, dtype=float)
    i = 0
    cost = []
    while True:
        i += 1
        res = -f(*x) + z
        cost.append(0.5 * np.dot(res, res))
        jac = j(*x)
        g = np.dot(jac.T, res)
        delta_x = np.linalg.solve(np.dot(jac.T, jac), g)
        x = x + delta_x * k
        if i > max_iter:
            break
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break

    cost = np.array(cost)
    return Result(nfev=i, cost=cost, grandnorm=np.linalg.norm(g), x=x)


def levenberg_marquardt(z, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4, max_iter=1000):
    x = np.asarray(x0, dtype=float)
    i = 0
    cost = []
    while True:
        i += 1
        res = -f(*x) + z
        cost.append(0.5 * np.dot(res, res))
        jac = j(*x)
        g = np.dot(jac.T, res)
        delta_x = np.linalg.solve(np.dot(jac.T, jac)+lmbd0*np.eye(np.dot(jac.T, jac).shape[0]), g)
        f1 = np.linalg.norm(f(*x))
        x = x + delta_x
        f2 = np.linalg.norm(f(*x))
        if f2 - f1 > 0:
            lmbd0 = lmbd0 * nu
        else:
            lmbd0 = lmbd0 / nu
        if i > max_iter:
            break
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, grandnorm=np.linalg.norm(g), x=x)
