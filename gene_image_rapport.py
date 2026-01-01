#%%
# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import math
from toy_signals import noisy_spikes_sine
from FocusFunctions import make_stftwin, A_to_max_focus, max_focus_to_A, rmin


#%%
# PARAMETERS

fs = 1000
t = np.linspace(-0.5, 0.5, fs)

signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

sigma = 0.05
window = np.exp(-(t**2) / (2*sigma**2))

t0 = 0.3
window_shifted = np.exp(-( (t - t0)**2 ) / (2*sigma**2))

windowed_signal = signal * window_shifted

#%%
# STFT EXPLICATION

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t, signal)
axs[0].set_title("Signal d’entrée")
axs[0].set_ylabel("Amplitude")
axs[0].axhline(0, color='black', linewidth=0.5)
axs[0].axvline(0, color='black', linewidth=0.5)

axs[1].plot(t, window_shifted)
axs[1].set_title("Fenêtre gaussienne à l'instant t=0.2")
axs[1].set_ylabel("Amplitude")
axs[1].axhline(0, color='black', linewidth=0.5)
axs[1].axvline(0, color='black', linewidth=0.5)

axs[2].plot(t, windowed_signal)
axs[2].set_title("Signal × Fenêtre translatée")
axs[2].set_xlabel("Temps en secondes")
axs[2].set_ylabel("Amplitude")
axs[2].axhline(0, color='black', linewidth=0.5)
axs[2].axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()

fou_tra = fft(signal)

fig, axs = plt.subplots(1, 2, figsize=(5, 10), sharey=True)

axs[0].plot(t, signal)
axs[0].set_title("Signal d’entrée")
axs[0].set_ylabel("Amplitude")

axs[1].plot(t, fou_tra)

#%%
# SIGNAL PARAMETERS

win_duration = 0.1
wintype = "gauss"
T = 1
Fs = 8000
nb_spikes = 20
psnr = 500
f1 = 1000
ampl = 1 / 20
times, x = noisy_spikes_sine(T, Fs, nb_spikes, psnr, f1, ampl)
alpha = 2
sigma_min = 0.1
A = 100
u = np.zeros(x.shape[0])
u[0] = 1
sigma = np.ones(x.shape[0])
winlen = math.floor(0.1 * 8000)
h = make_stftwin(0, "gauss", winlen, x.shape[0], sigma)
r = rmin(h, sigma_min, u, alpha, A)
nb_iter = 40

nb_points = 50
L = [1]
for i in range(1, nb_iter):
    r = rmin(h, sigma_min, u, alpha, i)
    L.append(A_to_max_focus(i, h, alpha, r))

plt.plot(L, label="Max_focus function")
plt.grid()
plt.xlabel("A")
plt.legend()
plt.show()

#%%
