#!/usr/bin/env python3


from mixfit import em_double_cluster
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
import json


center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
    column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='}, # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.0 * u.deg,
    height=1.0 * u.deg,
    catalog=['I/350'], # Gaia EDR3
)[0]
data = pd.read_csv('cluster.csv')
'''ra = stars['RAJ2000']._data   # прямое восхождение, аналог долготы
dec = stars['DEJ2000']._data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi) + ra.mean()
x2 = dec
v1 = stars['pmRA']._data
v2 = stars['pmDE']._data'''
ra = data['RAJ2000']
dec = data['DEJ2000']
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi) + ra.mean()
x2 = dec
v1 = data['pmRA']
v2 = data['pmDE']
x = np.vstack((x1, x2, v1, v2)).T
mu1 = np.mean(x1)
mu2 = np.mean(x2)
muv1 = np.mean(v1)
muv2 = np.mean(v2)
mu2 = mu1 = np.hstack([mu1, mu2])
muv = np.hstack([muv1, muv2])
sigmax2 = np.std(x1)
sigmav2 = np.std(v1)
sigma02 = np.std(v2)
mu1 = np.hstack([mu1, muv])
mu2 = np.hstack([mu2, muv])
sigma0 = np.zeros((2, 2))
sigma0[0, 0] = sigma0[1, 1] = sigma02
sigma = np.zeros((4, 4))
sigma[0, 0] = sigma[1, 1] = sigmax2
sigma[2, 2] = sigma[3, 3] = sigmav2
params = em_double_cluster(x, 0.2, 0.35, muv, mu1-5, mu2+5, sigma0, sigma)
size_ratio = round(params[0]/params[1], 2)
c1_ra = round(params[2][0], 2)
c1_dec = round(params[2][1], 2)
c2_ra = round(params[3][0], 2)
c2_dec = round(params[3][1], 2)
m_ra = round(params[3][2], 2)
m_dec = round(params[3][3], 2)
R1 = params[5][0][0]**0.5
R2 = params[5][1][1]**0.5
t1 = x[:, 0:2] - np.array([c1_ra, c1_dec])
red = np.where(np.linalg.norm(t1, axis=1) - R1 <= 0, np.array([x1, x2, v1, v2]), -1)
red_1 = np.delete(red, np.where(red == -1.0), axis=1)
t2 = x[:, 0:2] - np.array([c2_ra, c2_dec])
red = np.where(np.linalg.norm(t2, axis=1) - R2 <= 0, np.array([x1, x2, v1, v2]), -1)
red_2 = np.delete(red, np.where(red == -1.0), axis=1)

def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.20, 10.80), dpi=100) #загруженный график имеет dpi=300
    circle1 = plt.Circle((params[2][0], params[2][1]), params[5][0][0]**0.5, color='b', fill=False)
    circle2 = plt.Circle((params[3][0], params[3][1]), params[5][1][1]**0.5, color='r', fill=False)
    ax2.set_xlim((-5, 5))
    ax2.set_ylim((-5, 5))
    ax2.set_title('Собственное движение звёзд')
    ax2.set(xlabel='$v_{ra}$', ylabel='$v_{dec}$')
    ax2.scatter(v1, v2, s=0.7, color='g')
    ax2.scatter(red_1[2], red_1[3], s=0.7, c='b')
    ax2.scatter(red_2[2], red_2[3], s=0.7, c='r')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.set_title('Положения звёзд')
    ax1.set(xlabel='ra', ylabel='dec')
    ax1.scatter(x1, x2, s=0.7, color='g')
    ax1.scatter(red_1[0], red_1[1], s=0.7, color='b')
    ax1.scatter(red_2[0], red_2[1], s=0.7, color='r')
    plt.savefig('per.png')

plot()

def data():
    data = {
          "size_ratio": size_ratio,
          "motion": {"ra": m_ra, "dec": m_dec},
          "clusters": [
            {
              "center": {"ra": c1_ra, "dec": c1_dec},
            },
            {
              "center": {"ra": c2_ra, "dec": c2_dec},
            }
          ]
        }
    with open('per.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

data()


if __name__ == "__main__":
    pass
