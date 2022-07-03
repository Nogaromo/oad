import numpy as np

class lsq:

    def lstsq_ne(a, b):
        C = np.linalg.inv(a.T @ a) @ a.T
        x = C @ b
        cost = np.linalg.norm(b - a @ x) ** 2
        sigma = cost / (b.shape[0] - x.shape[0])
        var = np.linalg.inv((a.T @ a)) * sigma
        return (x, cost, var)

    def lstsq_svd(a, b, rcond=None):
        U, D, V = np.linalg.svd(a)
        D_1 = np.zeros(a.shape[0])
        for i in range(a.shape[1]):
            if D[i] != 0:
                D_1[i] = D[i]**(-2)
            else:
                D_1[i] = 0
        y = np.zeros(D.shape[0])
        z = U.T @ b
        if rcond is None:
            for i in range(D.shape[0]):
                if D[i] != 0:
                    y[i] = z[i] / D[i]
                else:
                    y[i] = 0
        else:
            for i in range(D.shape[0]):
                if D[i] != 0 and D[i] > rcond * max(D):
                    y[i] = z[i] / D[i]
                else:
                    y[i] = 0
        x = V.T @ y
        x = np.reshape(x, (a.shape[1], 1))
        cost = np.linalg.norm(b - a @ x) ** 2
        sigma = cost / (b.shape[0]-x.shape[0])
        var = V.T @ np.diag(D**(-2)) @ V * sigma
        return (x, cost, var)

    def lstsq(a, b, method, **kwargs):
        if method == "ne":
            return lsq.lstsq_ne(a, b)
        else:
            return lsq.lstsq_svd(a, b, rcond=kwargs.get("rcond"))

