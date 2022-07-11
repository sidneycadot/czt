#! /usr/bin/env -S python3 -B

import numpy as np
import matplotlib.pyplot as plt

from czt import czt_range

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

fft_1   = np.fft.rfft(y)
power_1 = np.abs(fft_1)**2
freq_1  = np.arange(len(power_1)) * fs / ns

plt.subplot(412)
plt.title("frequency domain using DFT ({} bins)".format(len(freq_1)))
plt.axvline(y_freq, c='r')
plt.plot(freq_1, power_1, "*-")
plt.xlim(10.0, 15.0)
plt.yscale('log')
plt.grid()

# Calculate and plot the high-resolution frequency domain signal using padding.

n_padding = 65536

fft_2   = np.fft.rfft(y, n_padding)
power_2 = np.abs(fft_2)**2
freq_2   = np.arange(len(power_2)) * fs / n_padding

plt.subplot(413)
plt.title("frequency domain using DFT with zero-padding for higher spectral resolution ({} bins on the full range)".format(len(freq_2)))
plt.axvline(y_freq, c='r')
plt.plot(freq_2, power_2, "*-")
plt.xlim(10.0, 15.0)
plt.yscale('log')
plt.grid()

# Calculate and plot a high-res frequency plot using the chirp-Z transform.

freq_min = 10.0
freq_max = 15.0
num_bins =  501

fft_3   = czt_range(y, num_bins, freq_min, freq_max, fs)
power_3 = np.abs(fft_3)**2
freq_3  = np.linspace(freq_min, freq_max, num_bins)

plt.subplot(414)
plt.title("frequency domain using Chirp-Z transform ({} bins on the range {} .. {} Hz)".format(len(freq_3), freq_min, freq_max))
plt.axvline(y_freq, c='r')
plt.plot(freq_3, power_3, "*-")
plt.xlim(freq_min, freq_max)
plt.yscale('log')
plt.grid()

# Show the combined plot.

plt.gcf().set_size_inches(12, 8)
plt.tight_layout()

plt.show()
