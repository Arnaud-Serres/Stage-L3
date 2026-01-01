import os
import sys

homedir = os.getenv("HOME")
sys.path.append(homedir + "/Science/git/focus/")

import numpy as np
from Transforms import tftft, fftft, make_stftwin


# %%
# TIME FOCUS


# Time focus
def time_focus(
    x, wintype, win_duration, a, Fs, max_focus, normalized=False, alpha=2, n=1
):
    """
    time_focus: compute a time focus function from a generic STFT: sum of
    fixed time slices of squared modulus, weighted by some power of the
    frequency, possibly normalized by the sums of fixed time slices of squared
    modulus
    usage: sigma = time_focus(x, wintype, win_duration, a, Fs, max_focus, normalized=False, n=1)


    Parameters
    ----------
    x : 1D array of float64
        Input signal
    wintype : Array of character
        Window type ("gaussian", "hann", "boxcar")
    win_duration : float64
        widow duration (in seconds)
    a : int
        time sampling step
    Fs : float64
        Sampling frequency
    max_focus : float64
        maximal value of the focus function
    normalized : boolean, optional
        DESCRIPTION. The default is False.
    alpha :  float64, optional
        Power of the STFT modulus to be used. The default is 2.
    n : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    sigma : array of float64
        Focus function.

    """
    L = x.shape[0]
    cst_focus = np.ones(L)

    # Compute STFT
    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)

    # Evaluate un-normalized focus function
    freqs = np.linspace(0, Fs / 2, W.shape[0])
    freqs = np.tile(freqs, (W.shape[1], 1)).T
    sigma = np.sum(freqs ** (alpha * n) * np.abs(W) ** alpha, axis=0)
    if normalized == True:
        sigma /= np.sum(np.abs(W) ** alpha, axis=0)

    # Normalization
    m = np.min(sigma)
    M = np.max(sigma)
    alpha = (max_focus - 1) / (M - m)
    beta = 1 - alpha * m
    sigma = alpha * sigma + beta

    return sigma


# Weight based time focus
def time_focus_w(x, wintype, win_duration, a, Fs, max_focus):
    L = x.shape[0]
    cst_focus = np.ones(L)
    # Compute STFT
    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)
    # Evaluate focus function
    weights = np.sum(np.abs(W) ** 2, axis=1)
    weights /= np.sum(weights)
    weights = np.tile(weights, (W.shape[1], 1)).T  ###### check that
    sigma = np.sum(weights * np.abs(W) ** 2, axis=0)
    # Normalization
    m = np.min(sigma)
    M = np.max(sigma)
    alpha = (max_focus - 1) / (M - m)
    beta = 1 - alpha * m
    sigma = alpha * sigma + beta

    return sigma


# Shannon Entropy based time focus
def time_focus_entropy(x, wintype, win_duration, a, Fs, max_focus, alpha=2):
    """
    time_focus_entropy: evaluate a time focus function using a Shannon entropy
    on a STFT of the input signal
    usage: sigma = time_focus_entropy(x,wintype,win_duration,a,Fs,max_focus,alpha)

    Parameters
    ----------
    x : array of float64
        Input signal.
    wintype : array of characters
        Window type  ('gauss', 'hann', 'boxcar').
    win_duration : float
        Window duration parameter (in seconds).
    a : int
        time sampling step for the transform.
    Fs : float64
        Sampling frequency.
    max_focus : float64
        Maximum value for the focus function.
    alpha : float64, optional
        Power of the STFT modulus to be used. The default is 2.

    Returns
    -------
    sigma : array of float64
        Focus function.

    """
    L = x.shape[0]
    cst_focus = np.ones(L)

    # Compute STFT
    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)

    sigma = time_focus_entropy_ref(W, max_focus, alpha)

    return sigma


def time_focus_entropy_ref(W, max_focus, alpha=2):
    """
    time_focus_entropy_ref: compute a Shannon entropy-based focus function
    from a time-frequency transform
    usage: sigma = time_focus_entropy_ref(W,max_focus,alpha)

    Parameters
    ----------
    W : 2D array of complex
        time-frequency transform from which the focus function is computed
    max_focus : float64
        Maximum value for the focus function.
    alpha : float64, optional
        Power of the STFT modulus to be used. The default is 2.

    Returns
    -------
    sigma : 1D array of float64
        Focus function.

    """
    P = np.abs(W) ** alpha
    P[0, :] += np.max(P) / 100

    # Evaluate un-normalized focus function
    norm_cst = np.sum(P, axis=0)
    norm_cst = np.tile(norm_cst, (W.shape[0], 1))
    P /= norm_cst
    sigma = -np.sum(P * np.log(P), axis=0)

    # Normalization
    m = np.min(sigma)
    M = np.max(sigma)
    alpha = (max_focus - 1) / (M - m)
    beta = 1 - alpha * m
    sigma = alpha * sigma + beta
    return sigma


# Renyi entropy based time focus
def time_focus_renyi_entropy(x, wintype, win_duration, a, Fs, max_focus, p, r):
    """
    time_focus_renyi_entropy:

    Parameters
    ----------
    x : array of float64
        Input signal.
    wintype : array of characters
        Window type  ('gauss', 'hann', 'boxcar').
    win_duration : float
        Window duration parameter (in seconds).
    a : int
        time sampling step for the transform.
    Fs : float64
        Sampling frequency.
    max_focus : float64
        Maximum value for the focus function.
    max_focus : TYPE
        DESCRIPTION.
    p : float64
        Index for Renyi entropy.
    r : 1D array of floart64
        regularization parameter.

    Returns
    -------
    sigma : 1D array of float64
        Focus function.

    """
    x_norm = np.linalg.norm(x, ord=2)
    L = x.shape[0]
    cst_focus = np.ones(L)

    # Compute STFT
    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)
    P = np.abs(W) ** 2
    P[0, :] += r * x_norm**2

    # Evaluate un-normalized focus function
    norm_cst = np.sum(P, axis=0)
    norm_cst = np.tile(norm_cst, (W.shape[0], 1))
    P /= norm_cst
    sigma = np.log(np.linalg.norm(P, ord=p, axis=0))
    sigma *= p / (1 - p)

    # Normalization
    m = np.min(sigma)
    M = np.max(sigma)
    alpha = (max_focus - 1) / (M - m)
    beta = 1 - alpha * m
    sigma = alpha * sigma + beta

    return sigma


def time_focus_renyi(x, wintype, win_duration, a, Fs, A, p, u, r):
    """
    time_focus_renyi:
        r is set to rmin

    Parameters
    ----------
    x : array of float64
        Input signal.
    wintype : array of characters
        Window type  ('gauss', 'hann', 'boxcar').
    win_duration : float
        Window duration parameter (in seconds).
    a : int
        time sampling step for the transform.
    Fs : float64
        Sampling frequency.
    A : int
        Sets maximum value for the focus function.
    p : float64
        Index for Renyi entropy (strictly larger than 2)
    r : 1D array of float64
        regularization parameter.
    u : 1D array of float64
        Regularization function

    Returns
    -------
    sigma : 1D array of float64
        Focus function.

    """
    cst_focus = np.ones(x.shape[0])
    x_norm = np.linalg.norm(x, ord=2)
    u_norm_p = np.linalg.norm(u, ord=p)
    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)
    W_abs_car = np.abs(W) ** 2
    rf = r * (x_norm**2)

    # Computing the numerator
    M, N = W.shape
    numerator = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            numerator[m, n] = W_abs_car[m, n] + rf * u[m]

    # Computing the numerator
    denominator = np.zeros(N)
    for n in range(N):
        denominator[n] = (np.linalg.norm(W[:, n]) ** 2) + rf

    # Computing the density
    rho = np.zeros((M, N))
    for n in range(N):
        rho[:, n] = numerator[:, n] / denominator[n]

    # Rényi entropy of the density
    g = np.zeros(N)
    for n in range(N):
        g[n] = (1 / (1 - p)) * np.log(np.sum(rho[:, n] ** p))

    # Compute Ru
    Ru = (p / (1 - p)) * np.log(u_norm_p)
    sigma = 1 + A * (g - Ru)

    return sigma


def time_focus_renyi_init(W, A, p, u, r):
    """
    time_focus_renyi_init:
        r is set to rmin
        Initializes the value of the focus function for the inversion scheme.
        Takes W a time-frequency transform as input instead of a signal like time_focus_renyi.

    Parameters
    ----------
    W : ndarray
        Time-frequency transform.
    A : int
        Sets maximum value for the focus function.
    p : float64
        Index for Renyi entropy (strictly larger than 2)
    r : 1D array of float64
        regularization parameter.
    u : 1D array of float64
        Regularization function

    Returns
    -------
    sigma : 1D array of float64
        Focus function.

    """
    W_norm = np.linalg.norm(W, ord=2)
    u_norm_p = np.linalg.norm(u, ord=p)
    W_abs_car = np.abs(W) ** 2
    rf = r * (W_norm**2)
    L, N = W.shape

    # Computing the numerator
    numerator = np.zeros((L, N))
    for k in range(L):
        for n in range(N):
            numerator[k, n] = W_abs_car[k, n] + rf * u[k]

    # Computing the denominator
    denominator = np.zeros(N)
    for n in range(N):
        denominator[n] = (np.linalg.norm(W[:, n]) ** 2) + rf

    # Computing the density
    rho = np.zeros((L, N))
    for n in range(N):
        rho[:, n] = numerator[:, n] / denominator[n]

    # Rényi entropy of the density
    g = np.zeros(N)
    for n in range(N):
        g[n] = (1 / (1 - p)) * np.log(np.sum(rho[:, n] ** p))

    # Compute sigma
    Ru = (p / (1 - p)) * np.log(u_norm_p)
    sigma = 1 + A * (g - Ru)
    return sigma


def rmin(h, sigma_min, u, p, A):
    """
    rmin:

    Parameters
    ----------
    h : array of float64
        Input signal.
    sigma_min : float
        Minimal value wanted of sigma
    p : float64
        Index for Renyi entropy (strictly larger than 2)
    u : 1D array of float64
        Regularization function
    A : Sets maximum value for the focus function.

    Returns
    -------
    rmin : array of float64

    """
    h_norm_inf = np.max(np.abs(h))
    h_norm_2 = np.linalg.norm(h, ord=2)
    u_norm_p = np.linalg.norm(u, ord=p)
    factor1 = (A * p) / ((p - 1) * (1 - sigma_min))
    return factor1 * (
        ((h_norm_inf ** (2 / p)) * (h_norm_2 ** (2 - (2 / p)))) / (u_norm_p)
    )


def max_focus_to_A(
    max_focus, p, h, sigma_min, u, precision=1e-5, A_inf=0, A_sup=1000, max_iter=1000
):
    """
    max_focus_to_A:
        Does a dichotomic search to find the value of A from sigma_max expression
        Returns the A associated with max_focus, with a precision of "precision"

    Parameters
    ----------
    max_focus : float
        Target value.
    h : array of float64
        Input signal.
    sigma_min : float
        Minimal value wanted of sigma
    p : float64
        Index for Renyi entropy (strictly larger than 2)
    u : 1D array of float64
        Regularization function
    precision : float
        Wanted precision of the result.
    A_inf : float
        Bottom value of A.
    A_sup : float
        Upper value of A.
    Max_iter : int
        Maximum iteration.

    Returns
    -------
    approx_A : float
        Approximated value of A, with a precision of "precision".

    """
    h_norm_2 = np.linalg.norm(h, ord=2)
    approx_A = 0.5 * (A_inf + A_sup)
    for i in range(max_iter):
        r = rmin(h, sigma_min, u, p, approx_A)
        produit = 1 + (((approx_A * p) / (p - 1)) * np.log(1 + (((h_norm_2) ** 2) / r)))
        if abs(produit - max_focus) < precision:
            return approx_A
        if produit - max_focus < 0:
            A_inf = approx_A
            approx_A = 0.5 * (A_inf + A_sup)
        else:
            A_sup = approx_A
            approx_A = 0.5 * (A_inf + A_sup)
    return approx_A


def A_to_max_focus(A, h, p, r):
    """
    A_to_max_focus:
        Returns the value of sigma_max for a given A

    Parameters
    ----------
    A : where to evaluate sigma_max
    h, p, r : necessary values to compute sigma_max

    Returns
    -------
    approx_A : float
        The value of sigma_max corresponding to the A given

    """
    h_norm_2 = np.linalg.norm(h, ord=2)
    return 1 + (((A * p) / (p - 1)) * np.log(1 + (((h_norm_2) ** 2) / r)))


# %%
# FREQUENCY FOCUS


# Entropy based frequency focus
def frequency_focus_entropy(x, n, scales, sigma_ref, sigma_max, alpha=1):
    """
    frequency_focus_entropy: Compute a frequency focus function using a
    Shannon entropy
    usage: sigma = frequency_focus_entropy(x, n, scales, sigma_ref, sigma_max, alpha=1)

    Parameters
    ----------
    x : 1D array of float64
        Input signal.
    n : int
        Derivative order of DOG wavelet.
    scales : array of float64
        scales at which the transform is computed
    sigma_ref : float64
        reference scale factor.
    sigma_max : float64
        Maximum value for the focus function.
    alpha : float64, optional
        Power of the STFT modulus to be used. The default is 2.

    Returns
    -------
    sigma : 1D array of float64
        Focus function.

    """
    nb_scales = len(scales)
    cst_focus = np.ones(nb_scales) * sigma_ref

    # Compute CWT
    W = fftft(x, n, scales, cst_focus, 1)
    P = np.abs(W) ** alpha

    # Evaluate un-normalized focus function
    norm_cst = np.sum(P, axis=1)
    norm_cst = np.tile(norm_cst, (W.shape[1], 1)).T
    P /= norm_cst
    sigma = -np.sum(P * np.log(P), axis=1)

    # Normalization
    m = np.min(sigma)
    M = np.max(sigma)
    alpha = (sigma_max - 1) / (M - m)
    beta = 1 - alpha * m
    sigma = alpha * sigma + beta

    return sigma
