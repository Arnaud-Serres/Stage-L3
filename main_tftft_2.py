import os
import sys

# homedir = os.getenv("HOME")
# sys.path.append(homedir + '/Science/git/focus/')

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
import math

from scipy.io import wavfile

from FocusFunctions import time_focus_entropy
from Transforms import tftft, itftft
from graphics import tfplot
from toy_signals import noisy_spikes_sine
from fonction_focus import (
    time_focus_renyi_init,
    time_focus_renyi_ite,
    make_stftwin_bis,
    max_focus_to_A,
)

# %%
# =============================================================================
# Global parameters
# =============================================================================

# Paths
respath = os.path.join(parent_dir, "results")
sigpath = os.path.join(parent_dir, "signals")
os.makedirs(respath, exist_ok=True)

# Spectrogram
alpha = 2

# Graphics
graph_file_flag = True
graph_file_ext = ".eps"

# Graphical parameters
cmap = "hot"
dynrange = 60


# %%
# =============================================================================
# Noisy spikes signal
# =============================================================================


# General parameters
signame = "noisy_spikes"
T = 1
Fs = 8000
nb_spikes = 20
psnr = 500
f1 = 1000
ampl = 1 / 20
times, x = noisy_spikes_sine(T, Fs, nb_spikes, psnr, f1, ampl)

# Window choice
wintype = "gauss"

# Time subsampling step (in time-frequency domain)
win_duration = 0.1  # (10 ms)
a = 4

# Parameters

# Focus function
sigma_max = 5
sigma_min = 0.5
cst_focus = np.ones(x.shape[0])
A = 10
u = np.zeros(x.shape[0])
u[0] = 1

sigma = time_focus_entropy(x, wintype, win_duration, a, Fs, sigma_max, alpha)
sigma_test = time_focus_renyi_ite(
    x, wintype, win_duration, a, Fs, A, alpha, u, sigma_min, cst_focus
)
fig, axs = plt.subplots(3)
axs[0].plot(times, x)
axs[0].grid()
axs[1].plot(np.linspace(0, 1, len(sigma)), sigma)
axs[1].grid()
axs[2].plot(np.linspace(0, 1, len(sigma_test)), sigma_test)
axs[2].grid()
# if graph_file_flag:
#     fname = respath + signame + '_tfoc' + graph_file_ext
#     fig.savefig(fname)
plt.show()

# # Unfocused and focused transforms
# V_u = tftft(x, wintype, win_duration, cst_focus, a, Fs=Fs)
# V_f = tftft(x, wintype, win_duration, sigma_test, a, Fs=Fs)

# # Display transforms
# fig2 = plt.figure()
# coef = tfplot(np.abs(V_u), a, np.array([0, 1]), fs=Fs, display=False,
#               dynrange=dynrange)
# _ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest',
#                aspect='auto', origin='lower', cmap=cmap)
# _ = plt.colorbar()
# _ = plt.xlabel('Time (sec.)')
# _ = plt.ylabel('Frequency (Hz)')
# # if graph_file_flag:
# #     fname = respath + signame + '_stft' + graph_file_ext
# #     fig2.savefig(fname)
# plt.show()

# fig3 = plt.figure()
# coef = tfplot(np.abs(V_f), a, np.array([0, 1]), fs=Fs, display=False, dynrange=60)
# _ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest',
#                aspect='auto', origin='lower',cmap=cmap)
# _ = plt.colorbar()
# _ = plt.xlabel('Time (sec.)')
# _ = plt.ylabel('Frequency (Hz)')
# if graph_file_flag:
#     fname = respath + signame + '_tftft' + graph_file_ext
#     fig3.savefig(fname)
# plt.show()

# # Check inversion
# x_rec = itftft(V_f, wintype, win_duration, sigma_test, len(x), Fs)
# fig4, axs4 = plt.subplots(2)
# axs4[0].plot(times, x, label="Original")
# axs4[0].grid()
# axs4[0].legend()
# axs4[1].plot(times, x_rec, label="Reconstructed")
# axs4[1].grid()
# axs4[1].legend()
# plt.show()

# %%
# Compute a focus function from the focused transform

# sigma2 = time_focus_entropy_ref(V_f,sigma_max,2)
# fig5,axs5 = plt.subplots(2)
# axs5[0].plot(np.linspace(0, 1, len(sigma)), sigma,label="True")
# axs5[0].grid()
# axs5[0].legend()
# axs5[1].plot(np.linspace(0, 1, len(sigma2)), sigma2,label="Approximate")
# axs5[1].grid()
# axs5[1].legend(loc = "lower left")
# plt.show()

winlen = math.floor(win_duration * Fs)

sigma0 = time_focus_renyi_init(
    x, wintype, win_duration, a, Fs, sigma_max, alpha, u, sigma_min
)
h0 = make_stftwin_bis(0, wintype, winlen, x.shape[0], sigma0)
A = max_focus_to_A(sigma_max, alpha, h0, sigma_min, u)
M = tftft(x, wintype, win_duration, sigma0, a, Fs)
nb_iter = 10

sigma_i = time_focus_renyi_ite(
    x, wintype, win_duration, a, Fs, A, alpha, u, sigma_min, sigma0
)
rec_err = np.zeros(nb_iter + 1)
for it in range(nb_iter):
    x_rec_i = itftft(M, wintype, win_duration, sigma_i, len(x), Fs)
    sigma_i = time_focus_renyi_ite(
        x_rec_i, wintype, win_duration, a, Fs, A, alpha, u, 0.5, sigma_i
    )
    rec_err[it] = np.linalg.norm(x_rec_i - x, ord=2) / np.linalg.norm(x, ord=2)

x_rec_end = itftft(M, wintype, win_duration, sigma_i, len(x), Fs)
rec_err[nb_iter] = np.linalg.norm(x_rec_end - x, ord=2) / np.linalg.norm(x, ord=2)

plt.semilogy(rec_err, label="Reconstruction error")
plt.grid()
plt.xlabel("Iteration")
plt.legend()
plt.show()


fig, axs = plt.subplots(3)
axs[0].plot(times, x)
axs[0].grid()
axs[1].plot(np.linspace(0, 1, len(sigma_test)), sigma_test)
axs[1].grid()
axs[2].plot(np.linspace(0, 1, len(sigma_i)), sigma_i)
axs[2].grid()
plt.show()
