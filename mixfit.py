#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    from scipy.optimize import minimize

    def func(p):
        tau, mu1, mu2, sigma1, sigma2 = p
        tau0 = tau
        tau1 = 1 - tau
        T1 = tau0 / np.sqrt(2 * np.pi * sigma1) * np.exp(-0.5 * (x - mu1) ** 2 / sigma1)
        T2 = tau1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-0.5 * (x - mu2) ** 2 / sigma2)
        T = np.sum(np.log(T1 + T2))
        return -T

    start = np.array([tau, mu1, sigma1, mu2, sigma2])
    param = minimize(func, x0=start, tol=rtol, method='Nelder-Mead')
    return param.x


def em_double_gauss(x: np.array, tau: float, mu1: float, mu2: float,
                    sigma1: float, sigma2: float, r_tol: float = 1e-3):
    def e(x: np.array, tau: float, mu1: float, mu2: float,
          sigma1: float, sigma2: float) -> np.array:
        tau0 = tau
        tau1 = 1 - tau
        T1 = tau0 / np.sqrt(2 * np.pi * sigma1) * np.exp(-0.5 * (x - mu1) ** 2 / sigma1)
        T2 = tau1 / np.sqrt(2 * np.pi * sigma2) * np.exp(-0.5 * (x - mu2) ** 2 / sigma2)
        T = T1 + T2

        T1 = np.divide(T1, T, out=np.full_like(T, 0.5), where=(T != 0))
        T2 = np.divide(T2, T, out=np.full_like(T, 0.5), where=(T != 0))
        return np.vstack((T1, T2))

    def m(x: np.array, *old):
        T1, T2 = e(x, *old)
        tau = np.sum(T1) / np.sum(T1 + T2)
        mu1 = np.sum(x * T1) / np.sum(T1)
        mu2 = np.sum(x * T2) / np.sum(T2)
        sigma1 = np.sum((x - mu1) ** 2 * T1) / np.sum(T1)
        sigma2 = np.sum((x - mu2) ** 2 * T2) / np.sum(T2)
        return tau, mu1, mu2, sigma1, sigma2

    th = (tau, mu1, mu2, sigma1, sigma2)
    diff = 1
    for i in range(100):
        th_new = m(x, *th)
        diff = np.abs(np.min((np.asarray(th_new) - np.asarray(th)) / np.asarray(th)))
        if diff < r_tol:
            break
        th = th_new
    return (th[0], th[1], th[2], th[3] ** 0.5, th[4] ** 0.5)


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma0, sigma, rtol=1e-5):
    from numpy.linalg import det

    def e(x, tau1, tau2, mu1, mu2, sigma0, sigma):

        tau3 = 1 - tau1 - tau2

        T1 = tau1 / np.sqrt((2 * np.pi) ** 4 * det(sigma)) * np.exp(-0.5 * np.dot(x - mu1,
                                                                                  np.linalg.inv(sigma) @ (
                                                                                              x - mu1).T).diagonal())

        T2 = tau2 / np.sqrt((2 * np.pi) ** 4 * det(sigma)) * np.exp(-0.5 * np.dot(x - mu2,
                                                                                  np.linalg.inv(sigma) @ (
                                                                                              x - mu2).T).diagonal())

        T3 = tau3 / np.sqrt((2 * np.pi) ** 2 * det(sigma0)) * np.exp(-0.5 * np.dot(x[:, 2:],
                                                                                   np.linalg.inv(sigma0) @ x[:,
                                                                                                           2:].T).diagonal())

        T = T1 + T2 + T3

        T1 = np.divide(T1, T, out=np.full_like(T, 0.5), where=(T != 0))
        T2 = np.divide(T2, T, out=np.full_like(T, 0.5), where=(T != 0))
        T3 = np.divide(T3, T, out=np.full_like(T, 0.5), where=(T != 0))

        return T1, T2, T3

    def m(x, *old):
        T1, T2, T3 = e(x, *old)

        tau1 = np.sum(T1) / np.sum(T1 + T2 + T3)
        tau2 = np.sum(T2) / np.sum(T1 + T2 + T3)
        mu1 = (x.T @ T1) / np.sum(T1)
        mu2 = (x.T @ T2) / np.sum(T2)
        t1 = np.vstack([T1, T1, T1, T1]).T
        t2 = np.vstack([T2, T2]).T
        sigma = (((x - mu1)).T @ ((x - mu1) * t1)) / np.sum(T1)
        sigma0 = ((((x[:, 2:]).T @ (x[:, 2:] * t2)))) / np.sum(T2)
        # по сути раскрытое внешнее произведение
        return tau1, tau2, mu1, mu2, sigma0, sigma

    th = (tau1, tau2, mu1, mu2, sigma0, sigma)
    diff = 0
    for i in range(500):
        th_new = m(x, *th)
        diff_old = diff
        diff = np.abs(np.linalg.norm(((np.asarray(th_new)[2]))) - np.linalg.norm(((np.asarray(th)[2]))))
        if abs(diff - diff_old) < rtol:
            break
        th = th_new

    return (th[0], th[1], th[2], th[3], th[4], th[5])


if __name__ == "__main__":
    pass
