import matplotlib.pyplot as plt
from opt import gauss_newton, levenberg_marquardt
import numpy as np
from scipy.integrate import quad
import pandas as pd
import json


def raw_to_norm():
    df_raw = pd.read_csv("/home/nogaromo/Загрузки/jla_mub.txt", delimiter=' ')
    df = df_raw.copy()
    df['mu'] = df['z']
    df['z'] = df['#']
    df.drop('#', axis=1)
    return df


def f(z, H0, omega, c=3.0 * 10 ** 11):
    int = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        int[i] = 5 * np.log10(
            c / H0 * (1 + z[i]) * quad(lambda x: 1 / np.sqrt((1 - omega) * (1 + x) ** 3 + omega), 0, z[i])[0]) - 5
    return int


def j(z, H0, omega, c=3.0 * 10 ** 11):
    jac = np.empty((z.size, 2), dtype=float)
    for i in range(z.shape[0]):
        integral = c / H0 * (1 + z[i]) * quad(lambda x: 1 / np.sqrt((1 - omega) * (1 + x) ** 3 + omega), 0, z[i])[0]
        jac[i, 0] = -5 / (H0 * np.log(10))
        jac[i, 1] = 5 * c * (1 + z[i]) / (H0 * np.log(10) * integral) * \
                    quad(lambda x: 0.5 * ((1 + x) * 3 - 1) / np.sqrt((1 - omega) * (1 + x) ** 3 + omega) ** (3 / 2), 0,
                         z[i])[0]
    return jac


z = raw_to_norm()['z']
mu = raw_to_norm()['mu']


r = gauss_newton(mu,
                 lambda *args: f(z, *args),
                 lambda *args: j(z, *args),
                 (50.0, 0.5),
                 k=0.01,
                 tol=1e-5,
                 max_iter=10000)

k = levenberg_marquardt(mu,
                 lambda *args: f(z, *args),
                 lambda *args: j(z, *args),
                 (50.0, 0.5),
                 lmbd0=0.01,
                 nu=1.5,
                 tol=1e-5,
                 max_iter=10000)


def data():
    data = {
        "Gauss-Newton": {"H0": np.round(r.x[0], 2), "Omega": np.round(r.x[1], 2), "nfev": r.nfev},
        "Levenberg-Marquardt": {"H0": np.round(k.x[0], 2), "Omega": np.round(k.x[1], 2), "nfev": k.nfev}
    }

    with open('parameters.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


data()


def graph_1():
    plt.figure(figsize=(19.20, 10.80))
    plt.plot(z, f(z, r.x[0], r.x[1]), color='#DC143C', label='model curve')
    plt.plot(z, mu, color='#13BC51', label='$\mu(z)$')
    plt.legend()
    plt.grid()
    plt.xlabel('z')
    plt.ylabel('$\mu$')
    plt.title('Зависимость красного свечения от расстояния')
    plt.savefig('mu-z.png', dpi=100)  #Загружал картинки с dpi=1200


graph_1()


def graph_2():
    plt.figure(figsize=(19.20, 10.80))
    plt.plot(np.arange(1, r.nfev + 1, 1), r.cost, color='#DC143C', label='Gauss-Newton')
    plt.plot(np.arange(1, k.nfev + 1, 1), k.cost, color='#13BC51', label='Levenberg-Marquardt')
    plt.legend()
    plt.grid()
    plt.xlabel('i')
    plt.ylabel('cost')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Зависимость функции потерь от итерационного шага')
    plt.savefig('cost.png', dpi=100)   #Загружал картинки с dpi=1200


graph_2()
