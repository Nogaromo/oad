import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import json


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


hdul = fits.open('/home/nogaromo/Загрузки/ccd.fits')
data = hdul[0].data
hdul.close()
std_1 = np.mean((data[0][0]-data[0][1])**2)
mean_1 = np.mean(data[0])
mean = np.zeros(100)
std = np.zeros(100)
for i in range(1, 100, 1):
    mean[i] = np.mean(data[i]) - mean_1
    std[i] = np.mean((data[i][0]-data[i][1])**2) - std_1
A = np.zeros((mean.shape[0], 2))
std = std.reshape((100, 1))
A[:, 0] = mean[:]
A[:, 1] = 1
koefs = lsq.lstsq(A, std, method='ne')[0]
plt.figure(figsize=(19.20, 10.80))
plt.plot(mean, std, label='std(mean)')
plt.plot(np.linspace(0, 1750, 100), koefs[0]*np.linspace(0, 1750, 100)+koefs[1], label='linear approx')
plt.grid()
plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
plt.savefig('/home/nogaromo/Загрузки/ccd.png', dpi=100)
a = koefs[0][0]
b = koefs[1][0]
g = 2 / a
sigma_r = g**2 * b / 2
delta_a = lsq.lstsq(A, std, method='ne')[2][0][0]
delta_b = lsq.lstsq(A, std, method='ne')[2][1][1]
dg = 2 / a**2 * np.sqrt(delta_a)
d_s = g**2 * np.sqrt(delta_b) / 2
'''dg на 2 порядка меньше d_s, поэтому ей пренебрегаем'''
s1 = np.sqrt(2*delta_b)/2*g*1/(2*np.sqrt(b))

data = {
    "ron": sigma_r.round(2),
    "ron_err": s1.round(2),
    "gain": g.round(2),
    "gain_err": dg.round(2)
}
with open('/home/nogaromo/Загрузки/ccd.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
file.close()
