#! /usr/bin/env -S python3 -B

import numpy as np
import matplotlib.pyplot as plt

from czt import czt

y_freq = 12.3456789 # signal frequency [Hz]

fs = 1000.0  # sample frequency [Hz]
ns = 800     # number of samples [-]

# Calculate and plot time the domain signal.

t = np.arange(ns) / fs
y = np.sin(t * y_freq * 2 * np.pi)

plt.subplot(411)
plt.title("time domain ({} samples)".format(len(y)))
plt.plot(t, y, '.')

# Calculate and plot the default-resolution frequency domain signal.

fft_y1 = np.fft.rfft(y)
power_y1 = np.abs(fft_y1)**2
freq1 = np.arange(len(power_y1)) * fs / ns

plt.subplot(412)
plt.title("frequency domain using DFT ({} bins)".format(len(freq1)))
plt.axvline(y_freq, c='r')
plt.plot(freq1, power_y1, "*-")
plt.xlim(10.0, 15.0)
plt.yscale('log')
plt.grid()

# Calculate and plot the high-resolution frequency domain signal using padding.

n_padding = 65536
fft_y2 = np.fft.rfft(y, n_padding)
power_y2 = np.abs(fft_y2)**2
freq2 = np.arange(len(power_y2)) * fs / n_padding

plt.subplot(413)
plt.title("frequency domain using DFT with zero-padding for higher spectral resolution ({} bins on the full range)".format(len(freq2)))
plt.axvline(y_freq, c='r')
plt.plot(freq2, power_y2, "*-")
plt.xlim(10.0, 15.0)
plt.yscale('log')
plt.grid()

# Calculate and plot a high-res frequency plot using the chirp-Z transform.

freq_min = 10.0
freq_max = 15.0
m        =  501  # number of bins.

w = np.exp(-(freq_max - freq_min) / ((m - 1) * fs) * 2 * np.pi * 1j)  # frequency step
a = np.exp(             freq_min  /            fs  * 2 * np.pi * 1j)  # first frequency

fft_y3 = czt(y, m, w, a)
power_y3 = np.abs(fft_y3)**2
freq3= np.linspace(freq_min, freq_max, m)

plt.subplot(414)
plt.title("frequency domain using Chirp-Z transform ({} bins on the range {} .. {} Hz)".format(len(freq3), freq_min, freq_max))
plt.axvline(y_freq, c='r')
plt.plot(freq3, power_y3, "*-")
plt.xlim(freq_min, freq_max)
plt.yscale('log')
plt.grid()

# Show the combined plot.

plt.gcf().set_size_inches(12, 8)
plt.tight_layout()

plt.show()
