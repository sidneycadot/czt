Chirp-Z implementation in Python
================================

This repository provides a Chirp-Z transform in Python.

The Chirp-Z transform provides a way to sample the frequency domain representation
of a discrete signal. Rather than sampling the entire frequency domain (as the FFT
does), it allows sampling of a sub-range, with a configurable frequency resolution.

The Chirp-Z transform can be implemented using three "normal" FFT invocations.

The Chirp-Z transform is an interesting alternative to using zero-padding to increase
the spectral resolution of the frequency domain. It uses a lot less memory, needing
only FFTs with size (m + n - 1) or greater. Our implementation selects the smallest
power-of-two that is greater than or equal to (m + n - 1), as power-of-two FFTs
are particularly efficient.

The ChirpZ transform also can be used to calculate an N-point FFT when only a
power-of-two FFT is available. This provides a reasonably efficient way to calculate
FFTs for any integer value N.

