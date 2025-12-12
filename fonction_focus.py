import numpy as np
import math
from Transforms import tftft
from FocusFunctions import rmin
from scipy.signal.windows import hann, gaussian, boxcar


def time_focus_renyi(x, wintype, win_duration, a, Fs, A, p, u, r):
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


def time_focus_renyi_init(M, A, p, u, r):
    M_norm = np.linalg.norm(M, ord=2)
    u_norm_p = np.linalg.norm(u, ord=p)
    M_abs_car = np.abs(M) ** 2
    rf = r * (M_norm**2)
    L, N = M.shape

    # Computing the numerator
    numerator = np.zeros((L, N))
    for k in range(L):
        for n in range(N):
            numerator[k, n] = M_abs_car[k, n] + rf * u[k]

    # Computing the denominator
    denominator = np.zeros(N)
    for n in range(N):
        denominator[n] = (np.linalg.norm(M[:, n]) ** 2) + rf

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


def time_focus_renyi_ite(
    x, wintype, win_duration, a, Fs, A, p, u, sigma_min, sigma_actu
):
    """
    time_focus_renyi_ite:
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
    winlen = math.floor(win_duration * Fs)
    h = make_stftwin_bis(0, wintype, winlen, x.shape[0], sigma_actu)
    A_bis = max_focus_to_A(5, p, h, sigma_min, u)
    r = rmin(h, sigma_min, u, p, A)
    print(A_to_max_focus(A_bis, h, p, r), "max_focus")
    W = tftft(x, wintype, win_duration, np.ones(x.shape[0]), a, Fs)

    # Computing the numerator
    P0 = np.abs(W) ** 2
    M, N = P0.shape
    P = np.zeros((M, N))
    for m in range(M):
        P[m, :] = P0[m, :] + r * (x_norm**2) * u[m]

    # Computing the density with regularis ation
    rho = np.zeros((M, N))
    for n in range(N):
        s = np.sum(P[:, n])
        rho[:, n] = P[:, n] / s

    # Rényi entropy of the density
    g = np.zeros(N)
    for n in range(N):
        g[n] = (1 / (1 - p)) * np.log(np.sum(rho[:, n] ** p))

    # Compute Ru
    Ru = np.linalg.norm(u, ord=p)
    Ru = (p / (1 - p)) * np.log(Ru)

    sigma = 1 + A * (g - Ru)
    return sigma


def make_stftwin_bis(t, wintype, winlen, L, sigma):
    h = np.zeros(L)
    if wintype == "gauss":
        w = gaussian(winlen, winlen / 10)
    elif wintype == "hann":
        w = hann(winlen)
    else:
        w = boxcar(winlen)
    h[:winlen] = w * np.sqrt(sigma[:winlen])
    h = np.roll(h, t - round(winlen / 2))
    return h


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
