import numpy as np
import matplotlib.pyplot as plt
import math

from Transforms import tftft, itftft
from FocusFunctions import rmin, make_stftwin, time_focus_entropy
from toy_signals import noisy_spikes_sine
from fonction_focus import (
    time_focus_renyi,
    time_focus_renyi_init,
    time_focus_renyi_ite,
    make_stftwin_bis,
    max_focus_to_A,
    A_to_max_focus
)


# Signal

T = 1
Fs = 8000
nb_spikes = 20
psnr = 500
f1 = 1000
ampl = 1 / 20
times, x = noisy_spikes_sine(T, Fs, nb_spikes, psnr, f1, ampl)


# Parameters

alpha = 10
# sigma_max = 5
sigma_min = 0.1
cst_focus = np.ones(x.shape[0])
A = 5
u = np.zeros(x.shape[0])
u[0] = 1
nb_iter = 10

# Window choice
wintype = "gauss"

# Time subsampling step (in time-frequency domain)
win_duration = 0.1  # (10 ms)
a = 4

winlen = math.floor(win_duration * Fs)
cst_focus = np.ones(x.shape[0])
winlen = math.floor(win_duration * Fs)
h = make_stftwin(0, wintype, winlen, x.shape[0], cst_focus)
r = rmin(h, sigma_min, u, alpha, A)
print("rmin :", r)
# Calcul de sigma_f
sigma_f = time_focus_renyi(x, wintype, win_duration, a, Fs, A, alpha, u, r)
M = tftft(x, wintype, win_duration, sigma_f, a, Fs)

# Iterations d'inversion
sigma_i = time_focus_renyi_init(M, A, alpha, u, r)
rec_err = np.zeros(nb_iter + 1)
for it in range(nb_iter):
    x_rec_i = itftft(M, wintype, win_duration, sigma_i, len(x), Fs)
    sigma_i = time_focus_renyi(x_rec_i, wintype, win_duration, a, Fs, A, alpha, u, r)
    rec_err[it] = np.linalg.norm(x_rec_i - x, ord=2) / np.linalg.norm(x, ord=2)

print("Max de sigma :", max(sigma_i))
x_rec_end = itftft(M, wintype, win_duration, sigma_i, len(x), Fs)
rec_err[nb_iter] = np.linalg.norm(x_rec_end - x, ord=2) / np.linalg.norm(x, ord=2)

plt.semilogy(rec_err, label="Reconstruction error")
plt.grid()
plt.xlabel("Iteration")
plt.legend()
plt.show()


# Comparison of sigma for 1 and 10 iterations

sigma = time_focus_renyi_init(M, A, alpha, u, r)

fig, axs = plt.subplots(3)
plt.xlabel("Comparaison des sigmas pour 1 et 10 iterations")
axs[0].plot(times, x)
axs[0].grid()
axs[1].plot(np.linspace(0, 1, len(sigma)), sigma)
axs[1].grid()
axs[2].plot(np.linspace(0, 1, len(sigma_i)), sigma_i)
axs[2].grid()
plt.show()
