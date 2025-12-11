import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

# -----------------------------
# Paramètres du signal et de la fenêtre
# -----------------------------
fs = 1000
t = np.linspace(-0.5, 0.5, fs)  # temps centré autour de 0

# Signal d’entrée : un mélange de sinusoïdes
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# Fenêtre gaussienne centrée en 0
sigma = 0.05
window = np.exp(-(t**2) / (2*sigma**2))

# Translation de la fenêtre à t0
t0 = 0.2
window_shifted = np.exp(-( (t - t0)**2 ) / (2*sigma**2))

# Produit signal × fenêtre translatée
windowed_signal = signal * window_shifted

# -----------------------------
# FIGURE
# -----------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 1) Signal
axs[0].plot(t, signal)
axs[0].set_title("Signal d’entrée")
axs[0].set_ylabel("Amplitude")
axs[0].axhline(0, color='black', linewidth=0.5)
axs[0].axvline(0, color='black', linewidth=0.5)

# 2) Fenêtre initiale
axs[1].plot(t, window_shifted)
axs[1].set_title("Fenêtre gaussienne à l'instant t=0.2")
axs[1].set_ylabel("Amplitude")
axs[1].axhline(0, color='black', linewidth=0.5)
axs[1].axvline(0, color='black', linewidth=0.5)

# 4) Produit signal × fenêtre translatée
axs[2].plot(t, windowed_signal)
axs[2].set_title("Signal × Fenêtre translatée")
axs[2].set_xlabel("Temps (s)")
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
