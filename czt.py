"""Chirp-Z transform."""

import numpy as np

def czt(x, m = None, w = None, a = None):
    """Calculate the Chirp-Z transform of an input vector x.

    Parameters:
        x: time domain samples.
        m: The number of frequency domain samples in the result.
        a: The complex starting point (normally, a point on the complex unit circle).
        w: The complex ratio between points (normally, a point on the complex unix circle). Determines the frequency resolution.
    """

    n = len(x)

    if m is None: m = n
    if w is None: w = np.exp(-2j * np.pi / m)
    if a is None: a = 1.

    w_exponents_1 =   np.arange(    0   , n) ** 2 / 2.0    # n elements         [ 0 .. (n - 1) ]
    w_exponents_2 = - np.arange(-(n - 1), m) ** 2 / 2.0    # m + n - 1 elements [ -(n - 1) .. +(m - 1) ]
    w_exponents_3 =   np.arange(    0   , m) ** 2 / 2.0    # m elements         [ 0        ..  (m - 1) ]

    xx = x * a ** -np.arange(n) * w ** w_exponents_1

    # Determine next-biggest FFT of power-of-two.

    nfft = 1
    while nfft < (m + n - 1):
        nfft += nfft

    # Perform CZT.

    fxx = np.fft.fft(xx, nfft)

    ww = w ** w_exponents_2

    fww = np.fft.fft(ww, nfft)

    fyy = fxx * fww

    yy = np.fft.ifft(fyy, nfft)

    # Select output.

    yy = yy[n - 1 : m + n - 1]

    y = yy * w ** w_exponents_3

    return y


def czt_range(x, m, fmin, fmax, fs=None):
    """Calculate frequency domain samples for a frequency range [fmin .. fmax].
    
    This function provides an interface to the CZT functionality that covers most use cases.

    Parameters:
        x: time domain samples.
        m: The number of frequency domain samples in the result.
        fmin: The minimum frequency for which to calculate the frequency response;
        fmax: The maximum frequency for which to calculate the frequency response;
        fs: The sampling frequency of the time domain samples in x. If not specified, it is assumed that the fmin and fmax values are already scaled to fs.
    """

    if fs is not None:
        fmin /= fs
        fmax /= fs

    a = np.exp(         fmin            * 2 * np.pi * 1j)  # first frequency
    w = np.exp(-(fmax - fmin) / (m - 1) * 2 * np.pi * 1j)  # frequency step

    return czt(x, m, w, a)
