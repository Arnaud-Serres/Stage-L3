import os
import sys

# homedir = os.getenv("HOME")
# sys.path.append(homedir + '/Science/git/focus/')

import numpy as np
from Transforms import tftft, fftft, make_stftwin

# %%
# Time focus


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


# Renyi entropy based time focus: NEW
# TO BE FINISHED


def time_focus_renyi(x, wintype, win_duration, a, Fs, max_focus, p, u):
    """
    time_focus_renyi:
        r is set to rmin
        A is computed from max_focus

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
    x_norm = np.linalg.norm(x, ord=2)
    L = x.shape[0]
    cst_focus = np.ones(L)

    # Compute window
    winlen = math.floor(win_duration * Fs)
    h = make_stftwin(0, wintype, winlen, L, sigma)
    # Compute STFT
    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)

    # Compute density, includeing regularization
    r = rmin(wintype, win_duration, p)
    P = np.abs(W) ** 2
    P[0, :] += r * x_norm**2

    # Evaluate un-normalized focus function
    norm_cst = np.sum(P, axis=0)
    norm_cst = np.tile(norm_cst, (W.shape[0], 1))
    P /= norm_cst
    sigma = np.log(np.linalg.norm(P, ord=p, axis=0))  # To be modified
    sigma *= p / (1 - p)

    # Normalization
    m = np.min(sigma)
    M = np.max(sigma)
    alpha = (max_focus - 1) / (M - m)
    beta = 1 - alpha * m
    sigma = alpha * sigma + beta

    return sigma


def time_focus_renyi_test(x, wintype, win_duration, a, Fs, max_focus, p, u):
    """
    time_focus_renyi:
        r is set to rmin
        A is computed from max_focus

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
    x_norm = np.linalg.norm(x, ord=2)
    L = x.shape[0]
    cst_focus = np.ones(L)

    W = tftft(x, wintype, win_duration, cst_focus, a, Fs)

    # Compute density, including regularization
    r = rmin(wintype, win_duration, p)
    P = np.abs(W) ** 2
    P[0, :] += r * x_norm**2

    # Evaluate un-normalized focus function
    norm_cst = np.sum(P, axis=0)
    norm_cst = np.tile(norm_cst, (W.shape[0], 1))
    P /= norm_cst
    Rp = np.log(np.linalg.norm(P, ord=p, axis=0))
    Rp *= p / (1 - p)

    # Compute Ru
    Ru = np.linalg.norm(u, ord=p)
    Ru = (p / (p - 1)) * np.log(Ru)

    sigma = Rp - Ru

    max_sigma = np.max(sigma)

    A = compute_A(max_focus)  # ou max_sigma?

    sigma = 1 + A * sigma
    return sigma


def compute_A(max_focus, precision=1e-5, A_inf=0, A_sup=20, max_iter=1000):
    """
    compute_A:
        Does a dichotomic search on x --> x*ln(1 + 1/x)
        Returns A such that A*ln(1 + 1/A) is approximatly max_focus, with a precision of "precision"

    Parameters
    ----------
    max_focus : float
        Target value.
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
    approx_A = 0.5 * (A_inf + A_sup)
    for i in range(max_iter):
        if abs((approx_A * np.log(1 + (1 / approx_A))) - max_focus) < precision:
            return approx_A
        if ((approx_A * np.log(1 + (1 / approx_A))) - max_focus) < 0:
            A_inf = approx_A
            approx_A = 0.5 * (A_inf + A_sup)
        else:
            A_sup = approx_A
            approx_A = 0.5 * (A_inf + A_sup)
    return approx_A


def rmin(h, sigma_min, u, p, A):
    h_norm_inf = np.max(np.abs(h))
    h_norm_2 = np.linalg.norm(h, ord=2)
    u_norm_p = np.linalg.norm(u, ord=p)
    factor1 = (A * p) / ((p - 1)*(1 - sigma_min))
    return (
        factor1 * ((h_norm_inf ** (2 / p)) * (h_norm_2 ** (2 - (2 / p)))) / (u_norm_p)
    )


def A_cst(sigma_max, wintype, win_duration, kappa, p):
    A = 1


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
