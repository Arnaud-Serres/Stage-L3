#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2025

@author: bruno
"""

import numpy as np
import math
from scipy.fft import fft as fft, ifft as ifft, rfft as rfft, irfft as irfft
from scipy.signal.windows import hann, gaussian, boxcar

# %%

# =============================================================================
# WAVEFORMS & ATOMS
# =============================================================================


# Cauchy wavelets with squared frequencies
def sqcauchy_wavelet(L, n, scale, Fourier=False):
    """
    sqcauchy_wavelet: generate a Cauchy wavelet, in the Fourier domain
    usage: hat_psi = sqcauchy_wavelet(L,n,scale,Fourier=False)

    Parameters
    ----------
    L : int
        length of the analysis frame
    n : int
        number of vanishing moments.
    scale : float64
        scale at which the wavelet is computed
    Fourier : boolean, optional
        If set to True, the wavelet in frequency domain will be returned.
        The default is False.

    Returns
    -------
    hat_psi : array of complex
        wavelet in time domain (or frequency domain if Fourier = True)

    """
    L1 = int(np.floor(L / 2)) + 1
    s_min = 4 * np.sqrt(n)
    freqs = np.linspace(0, 1 / 2, L1)
    freqs *= scale
    freqs2 = freqs**2
    tmp = freqs2**n * np.exp(-freqs2 * s_min**2)
    tmp /= np.sqrt(np.sum(tmp**2))
    hat_psi = np.zeros(L)
    hat_psi[range(L1)] = tmp
    if Fourier == False:
        psi = ifft(hat_psi)
        return psi
    else:
        return hat_psi


# Frequency-focused atoms
def ff_atoms(L, n, scale, sigma, Fourier=False):
    """
    ff_atoms: generate frequency-focused time-frequency atoms in the
    fourier domain
    usage: = hat_psi = ff_atoms(L,n,scale,sigma):

    Parameters
    ----------
    L : int16
        length of the analysis frame
    n : int16
        number of vaishing moments.
    scale : float64
        scale at which the wavelet is computed
    sigma : float64
        focus parameter

    Returns
    -------
    hat_psi : 1d array of complex128
        Frequency focused wavelet (Fourier transform).

    """
    L1 = int(np.floor(L / 2)) + 1
    s_min = 4 * np.sqrt(n)
    # k_0 = int(np.floor(L/4))
    k_0 = 1 / 4
    freqs = np.linspace(0, 1 / 2, L1)
    freqs *= sigma / scale
    freqs -= (sigma - 1) * k_0
    freqs2 = freqs**2
    tmp = freqs2**n * np.exp(-freqs2 * s_min**2)
    tmp[np.where(freqs <= 0)] = 0
    tmp /= np.sqrt(np.sum(tmp**2))
    hat_psi = np.zeros(L)
    hat_psi[range(L1)] = tmp

    if Fourier == False:
        psi = ifft(hat_psi)
        return psi
    else:
        return hat_psi


# STFT window
def make_stftwin(t, wintype, winlen, L, sigma):
    """
    make_stftwin: compute focused stft at fixed time
    usage: h = make_stftwin(t,wintype,winlen,L,sigma)

    Parameters
    ----------
    t : int
        center time of the window.
    wintype : str
        Window type  ('gauss', 'hann', 'boxcar')
    winlen : int
        Support length of the window.
    L : int
        Window length.
    sigma : float
        local focus at time t.

    Returns
    -------
    h : array of float64
        window.

    """

    h = np.zeros(L)
    if wintype == "gauss":
        h[range(winlen)] = gaussian(winlen, winlen / 20)
    elif wintype == "hann":
        h[range(winlen)] = hann(winlen)
    else:
        h[range(winlen)] = boxcar(winlen)
    h *= np.sqrt(sigma)

    # Center the window at the appropriate location
    h = np.roll(h, t - round(winlen / 2))

    return h


# %%

# =============================================================================
# TIME-FOCUSED TRANSFORM
# =============================================================================


# Time-focused STFT
def tftft(x, wintype, win_duration, sigma, a=1, Fs=1):
    """
    tftft: tyime-focused time-frequency transform
    usage: x_rec = tftft(x, wintype, win_duration, sigma, a=1, Fs=1)

    Parameters
    ----------
    x : array of float64
        Input signal
    wintype : str
        Window type  ('gauss', 'hann', 'boxcar')
    win_duration : float
        Window duration parameter (in seconds)
    sigma : array of float64
        Focus function.
    a : Positive Integer, optional
        Time subsampling step. The default is 1.
    Fs : Float, optional
        Sampling frequency. The default is 1.

    Returns
    -------
    V : 2D array of complex
        Time-focused transform.

    """
    # Setting up parameters
    L = x.shape[0]
    win_length = math.floor(win_duration * Fs)
    N = math.floor(L / a)

    # Initialize transform
    W = np.zeros((math.floor(L / 2) + 1, N), dtype=complex)

    # Compute transform
    for n in range(N):
        t = math.floor(n * a)

        # generate window
        winlen_t = math.floor(win_length / sigma[n])
        h_t = make_stftwin(t, wintype, winlen_t, L, sigma[n])

        # Compute fixed time transform
        tmp = x * h_t
        W[:, n] = rfft(tmp, norm="ortho")
    return W


# Inverse time-focused STFT
def itftft(V, wintype, win_duration, sigma, L, Fs=1):
    """
    itftft: Inverse time-focused time-frequency transform, with known
    focus function
    usage: x_rec = itftft(V, wintype, win_duration, sigma, L, Fs=1)

    Parameters
    ----------
    V : 2D array of complex128
        Time-focused transform.
    wintype : str
        Window type  ('gauss', 'hann', 'boxcar')
    win_duration : float
        Window duration parameter (in seconds)
    sigma : array of float64
        Focus function.
    a : int
        Time subsampling step, optional. The default is 1. Must be >0
    Fs : float
        Sampling frequency, optional. The default is 1.

    Returns
    -------
    x_rec : array of float64
        Reconstructed signal

    """
    # Setting up parameters
    N = V.shape[1]
    win_length = math.floor(win_duration * Fs)
    a = math.floor(L / N)

    # Compute fixed time inverse Fourier transform
    V_invhat = irfft(V, n=L, axis=0, norm="ortho")

    # Initialize admissibility function
    k_sigma = np.zeros(L)
    for n in range(N):
        t = math.floor(n * a)

        # Generate window
        winlen_t = math.floor(win_length / sigma[n])
        h_t = make_stftwin(t, wintype, winlen_t, L, sigma[n])

        # Add window contribution
        V_invhat[:, n] *= h_t

        # Update admissibility fonction
        k_sigma += np.abs(h_t) ** 2

    # Compute inverse tf-stft
    x_rec = np.sum(V_invhat, axis=1) / k_sigma
    return x_rec


# %%

# =============================================================================
# FREQUENCY-FOCUSED TRANSFORMS
# =============================================================================


# Continuous wavelet transform
def cwt(x, n, scales, Fs=1):
    """
    cwt: continuous wavelet transform (complex valued, analytic DOG wavelets)
    usage: M = cwt(x,n,scales,Fs=1)

    Parameters
    ----------
    x : 1D array of float64
        Input signal
    n : int
        Number of vanishing moments
    scales : array of float64
        scales at which the transform is computed
    Fs :  Float, optional
        Sampling frequency. The default is 1.

    Returns
    -------
    M : 2D array of complex
        Continuous wavelet transform

    """
    L = len(x)
    hat_x = fft(x)
    nb_scales = len(scales)
    M = np.zeros((nb_scales, L), dtype=complex)
    s = 0
    for sc in scales:
        hat_psi = sqcauchy_wavelet(L, n, sc, Fourier=True)
        tmp = hat_psi * hat_x
        M[s, :] = ifft(tmp)
        s += 1
    return M


# Frequency-focused transform
def fftft(x, n, scales, sigma, Fs=1):
    """
    fftft: frequency focused continuous wavelet transform
    usage: M = fftft(x,n,scales,sigma,Fs=1)

    Parameters
    ----------
    x : 1D array of float64
        Input signal
    n : int
        Number of vanishing moments
    scales : array of float64
        scales at which the transform is computed
    sigma : array opf float64
        focus function
    Fs :  Float, optional
        Sampling frequency. The default is 1.

    Returns
    -------
    M : 2D array of complex
        Continuous wavelet transform

    """
    L = len(x)
    hat_x = fft(x)
    nb_scales = len(scales)
    M = np.zeros((nb_scales, L), dtype=complex)

    for k in range(nb_scales):
        hat_psi = ff_atoms(L, n, scales[k], sigma[k])
        tmp = hat_psi * hat_x
        M[k, :] = ifft(tmp)
    return M
