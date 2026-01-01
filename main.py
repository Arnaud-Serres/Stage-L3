#%% 
# IMPORTS

import os
import sys
from scipy.io import wavfile
import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
from graphics import tfplot


from Transforms import tftft, itftft
from FocusFunctions import (
    rmin,
    make_stftwin,
    time_focus_renyi,
    time_focus_renyi_init,
    max_focus_to_A,
    A_to_max_focus,
)
from toy_signals import noisy_spikes_sine

#%% 
# GLOBAL PARAMETERS

# Signal
T = 1
Fs = 8000
nb_spikes = 20
psnr = 500
f1 = 1000
ampl = 1 / 20

# Focus / inversion
alpha = 10
sigma_min = 0.1
A = 5
nb_iter = 10

# Window / TF
wintype = "gauss"
win_duration = 0.1  # (10 ms)
winlen = math.floor(win_duration * Fs)
a = 4

# Initialisations
u = None
cst_focus = None

#%% 
# LOADING SIGNAL

times, x = noisy_spikes_sine(T, Fs, nb_spikes, psnr, f1, ampl)

u = np.zeros(x.shape[0])
u[0] = 1


#%% 
# LOADING GLOCKENSPIEL SIGNAL
# Very heavy, any operation takes time

a = 8
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

fname = os.path.join(parent_dir, "signals/glock.wav")
signame = "glock"

Fs, glock = wavfile.read(fname)
T = len(glock) / Fs
times = np.linspace(0, T, len(glock))

subsampling = 4
glock2 = scipy.signal.decimate(glock, subsampling, 8)
Fs2 = Fs / subsampling

seg = range(40000)
x = glock2[seg]

T2 = len(x) / Fs2
times = np.linspace(0, T2, len(x))


#%% 
# BOUCLE D'INVERSION

# Initialisation
cst_focus = np.ones(x.shape[0])
u = np.zeros(x.shape[0])
u[0] = 1
h = make_stftwin(0, wintype, winlen, x.shape[0], cst_focus)
r = rmin(h, sigma_min, u, alpha, A)
print("rmin :", r)

# Computing sigma_f
sigma_f = time_focus_renyi(x, wintype, win_duration, a, Fs, A, alpha, u, r)

W = tftft(x, wintype, win_duration, sigma_f, a, Fs)
sigma_i = time_focus_renyi_init(W, A, alpha, u, r)
rec_err = np.zeros(nb_iter + 1)
for it in range(nb_iter):
    x_rec_i = itftft(W, wintype, win_duration, sigma_i, len(x), Fs)
    sigma_i = time_focus_renyi(x_rec_i, wintype, win_duration, a, Fs, A, alpha, u, r)
    rec_err[it] = np.linalg.norm(x_rec_i - x, ord=2) / np.linalg.norm(x, ord=2)

print("Max de sigma :", max(sigma_i))
x_rec_end = itftft(W, wintype, win_duration, sigma_i, len(x), Fs)
rec_err[nb_iter] = np.linalg.norm(x_rec_end - x, ord=2) / np.linalg.norm(x, ord=2)

#%% 
# ERREUR DE RECONSTRUCTION

plt.semilogy(rec_err, label="Reconstruction error")
plt.grid()
plt.xlabel("Iteration")
plt.legend()
plt.show()


#%% 
# FOCUS FUNCTIONS COMPARAISON

sigma = time_focus_renyi_init(W, A, alpha, u, r)

fig, axs = plt.subplots(3)
plt.xlabel("Comparing ")

axs[0].plot(times, x)
axs[0].grid()

axs[1].plot(np.linspace(0, 1, len(sigma)), sigma)
axs[1].grid()

axs[2].plot(np.linspace(0, 1, len(sigma_i)), sigma_i)
axs[2].grid()

for ax in axs:
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.yaxis.get_major_formatter().set_scientific(False)

plt.tight_layout()
plt.show()


#%% 
# CLASSIC STFT SPECTROGRAM

W_cst = tftft(x, wintype, win_duration, np.ones(x.shape[0]), a, Fs)
W_mag = np.abs(W_cst)

dynrange = 60
coef = 20 * np.log10(W_mag / np.max(W_mag))
coef = np.maximum(coef, -dynrange)

plt.figure()
plt.imshow(
    coef,
    aspect="auto",
    origin="lower",
    extent=(0, T, 0, Fs / 2),
    cmap="hot",
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label="Amplitude")
plt.show()


#%% 
# FOCUS SPECTROGRAM

W_sig = tftft(x, wintype, win_duration, sigma_i, a, Fs)

fig2 = plt.figure()
coef = tfplot(
    np.abs(W_sig), a, np.array([0, 1]), fs=Fs, display=False, dynrange=dynrange
)
plt.imshow(
    coef,
    extent=(0, T, 0, Fs / 2),
    interpolation="nearest",
    aspect="auto",
    origin="lower",
    cmap="hot",
)
plt.colorbar()
plt.xlabel("Time (sec.)")
plt.ylabel("Frequency (Hz)")
plt.show()

# %%
