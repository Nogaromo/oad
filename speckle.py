from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from photutils.detection import find_peaks
from astropy.stats import sigma_clipped_stats
import json


data = fits.open('speckledata.fits')[2].data
mean_data = np.mean(data, axis=0)

fig = plt.gcf()
fig.set_size_inches(5.12, 5.12)
fig.set_dpi(100)


def plot_mean():
    plt.imshow(mean_data, cmap='gray')
    plt.savefig('mean.png')


plot_mean()


def plot_fft():
    fft = np.fft.fft2(data)
    fft1 = np.fft.fftshift(fft)
    p = np.abs(fft1)**2
    p_mean = np.mean(p, axis=0)
    plt.imshow(p_mean, cmap='gray', vmin=0, vmax=np.quantile(p_mean, 0.98))
    plt.savefig('fourier.png')
    return p_mean


p_mean = plot_fft()


ones = np.ones((200, 200))
for i in range(200):
    for j in range(200):
        if np.sqrt((i-100)**2+(j-100)**2) <= 50.0:
            ones[i][j] = 0


def rot_plot():
    angles = np.linspace(0, 360, 721)
    rot = np.zeros((721, 200, 200))
    for i in range(721):
        rot[i] = scipy.ndimage.rotate(p_mean, angles[i], reshape=False)
    rot_mean = np.mean(rot, axis=0)
    plt.imshow(rot_mean, cmap="gray", vmin=0, vmax=np.quantile(rot_mean, 0.98))
    plt.savefig('rotaver.png')
    return rot_mean


cringe = rot_plot()


def last():
    vvv = p_mean / cringe
    temp = np.ma.masked_array(vvv, mask=ones).filled(0)
    ifft2 = np.fft.ifft2(temp)
    ifft2_shift = np.fft.ifftshift(ifft2)
    ifft22 = np.abs(ifft2_shift)
    plt.xlim(75, 125)
    plt.ylim(125, 75)
    plt.imshow(ifft22, cmap="gray", vmin=0, vmax=np.quantile(ifft22, 0.9998))
    plt.savefig('binary.png')
    return ifft22


iff = last()


mean, median, std = sigma_clipped_stats(iff, sigma=3.0)
threshold = median + (5. * std)
tbl = find_peaks(iff, threshold, npeaks=3)
dist_in_pixels = np.linalg.norm(np.array([tbl[1][0] - tbl[0][0], tbl[1][1] - tbl[0][1]]))
k = 0.0206
dist_in_sec = k * dist_in_pixels
data_json = {"distance": dist_in_sec.round(4)}
with open('binary.json', 'w', encoding='utf-8') as file:
    json.dump(data_json, file, ensure_ascii=False, indent=4)
