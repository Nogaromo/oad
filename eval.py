import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

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



N = 10000
A = np.random.rand(500, 20)
x_1 = np.random.rand(20)
mu = A @ x_1
r = np.zeros((500, 500))
for i in range(500):
    r[i][i] = 0.01
rng = np.random.default_rng()
cost_1 = np.zeros(N)
for i in range(N):
    b = rng.multivariate_normal(mu, r, size=1).T
    cost_1[i] = lsq.lstsq(A, b, method='ne')[1]
cost_2 = np.zeros(N)
for i in range(N):
    b = rng.multivariate_normal(mu, r, size=1).T
    cost_2[i] = lsq.lstsq(A, b, method='svd', **{"rcond": None})[1]
plt.subplots(figsize=(19.20, 10.80))
plt.hist(cost_1, color='blue', density=True, bins=50, label='cost dist via lsq')
plt.hist(cost_2, color='red', density=True, bins=50, label='cost dist via svd')
plt.plot(np.linspace(350, 600, 500)/100, chi2.pdf(np.linspace(350, 600, 500), 480)*100, 'o', color='green', label='$\chi_2$ dist')
plt.grid()
plt.title('$\chi_2$ and cost distribution')
plt.legend()
plt.savefig('/home/nogaromo/Загрузки/chi2.png', dpi=100)
plt.show()
