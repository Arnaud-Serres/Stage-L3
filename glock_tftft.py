#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:24:04 2025

@author: bruno
"""

import os
import sys

homedir = os.getenv("HOME")
sys.path.append(homedir + '/Science/git/focus/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
from scipy.io import wavfile

from FocusFunctions import (
    time_focus,
    time_focus_entropy,
    time_focus_renyi_entropy,
    time_focus_entropy_ref,
)
from Transforms import tftft, itftft
from graphics import tfplot


# %%
# =============================================================================
# Global parameters
# =============================================================================

# # Paths
refpath = homedir + '/Science/git/focus/'
respath = refpath + 'results/'
sigpath = refpath + 'signals/'

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
# Load and preprocess Glockenspiel signal
# =============================================================================

# load signal
fname = sigpath + 'glock.wav'
signame = 'glock'
Fs, glock = wavfile.read(fname)
T = len(glock)/Fs
times = np.linspace(0,T,len(glock))

# Subsample
subsampling = 4
glock2 = scipy.signal.decimate(glock, subsampling, 8)
Fs2 = Fs / subsampling

# shorten
seg = range(40000)
glock2 = glock2[seg]
T2 = len(glock2) / Fs2
times2 = np.linspace(0, T2, len(glock2))

# %%
# =============================================================================
# Time-frequency transforms
# =============================================================================

# Window choice
wintype = "gauss"

# Time subsampling step (in time-frequency domain)
win_duration = 0.1  # (10 ms)
a = 8

# Max focus
sigma_max = 5

# ## STANDARD FOCUSED FUNCTION
# # Compute standard focus function
# sigma = time_focus(glock2, wintype, win_duration, a, Fs2, sigma_max, False,
#                    alpha = alpha, n = 2)
# cst_focus = np.ones(sigma.shape)
# fig, axs = plt.subplots(2)
# axs[0].plot(times2, glock2)
# axs[0].grid()
# axs[1].plot(np.linspace(0,T2,len(sigma)),sigma)
# axs[1].grid()
# if graph_file_flag == True:
#     fname = respath + signame + '_tfoc' + graph_file_ext
#     fig.savefig(fname)
# plt.show()

# # Compute unfocused and focused transform
# V_u = tftft(glock2, wintype, win_duration, cst_focus, a, Fs2)
# V_f = tftft(glock2, wintype, win_duration, sigma, a, Fs2)

# # Display unfocused transform
# fig2 = plt.figure()
# coef = tfplot(np.abs(V_u), a, np.array([0, 1]), fs=Fs2, display=False,
#               dynrange=dynrange)
# _ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest',
#                aspect='auto', origin='lower',  cmap=cmap)
# _ = plt.colorbar()
# _ = plt.xlabel('Time (sec.)')
# _ = plt.ylabel('Frequency (Hz)')
# if graph_file_flag:
#     fname = respath + signame + '_stft' + graph_file_ext
#     fig2.savefig(fname)
# plt.show()

# # Display focused transform
# fig3 = plt.figure()
# coef = tfplot(np.abs(V_f), a, np.array([0, 1]), fs=Fs2, display=False,
#               dynrange=dynrange)
# _ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest',
#                aspect='auto', origin='lower', cmap = cmap)
# _ = plt.colorbar()
# _ = plt.xlabel('Time (sec.)')
# _ = plt.ylabel('Frequency (Hz)')
# if graph_file_flag:
#     fname = respath + signame + '_tftft' + graph_file_ext
#     fig3.savefig(fname)
# plt.show()

## ENTROPY-BASED FOCUSED FUNCTION
# Compute standard focus function
# sigma = time_focus_entropy(glock2, wintype, win_duration, a, Fs2, sigma_max, alpha=alpha)
r = 0.00001
p = 4
sigma = time_focus_renyi_entropy(glock2, wintype, win_duration, a, Fs2, sigma_max, p, r)
cst_focus = np.ones(sigma.shape)
fig, axs = plt.subplots(2)
axs[0].plot(times2, glock2)
axs[0].grid()
axs[1].plot(np.linspace(0, T2, len(sigma)), sigma)
axs[1].grid()
if graph_file_flag == True:
    fname = respath + signame + '_tfoc_ent' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Compute unfocused and focused transform
V_u = tftft(glock2, wintype, win_duration, cst_focus, a, Fs2)
V_f = tftft(glock2, wintype, win_duration, sigma, a, Fs2)

# Display unfocused transform
fig2 = plt.figure()
coef = tfplot(
    np.abs(V_u), a, np.array([0, 1]), fs=Fs2, display=False, dynrange=dynrange
)
_ = plt.imshow(
    coef,
    extent=(0, T2, 0, Fs2 / 2),
    interpolation="nearest",
    aspect="auto",
    origin="lower",
    cmap=cmap,
)
_ = plt.colorbar()
_ = plt.xlabel("Time (sec.)")
_ = plt.ylabel("Frequency (Hz)")
if graph_file_flag:
    fname = respath + signame + '_stft_ent' + graph_file_ext
    fig2.savefig(fname)
plt.show()

# Display focused transform
fig3 = plt.figure()
coef = tfplot(
    np.abs(V_f), a, np.array([0, 1]), fs=Fs2, display=False, dynrange=dynrange
)
_ = plt.imshow(
    coef,
    extent=(0, T2, 0, Fs2 / 2),
    interpolation="nearest",
    aspect="auto",
    origin="lower",
    cmap=cmap,
)
_ = plt.colorbar()
_ = plt.xlabel("Time (sec.)")
_ = plt.ylabel("Frequency (Hz)")
if graph_file_flag:
    fname = respath + signame + '_tftft_ent' + graph_file_ext
    fig3.savefig(fname)
plt.show()

# %%

# =============================================================================
# Display focused windows
# =============================================================================

# Display 3 sample windows
L_red = 1500
times_red = times2[range(L_red * a)]
t_red_max = max(times_red)
sigma_red = sigma[range(L_red)]
toto = np.argsort(-sigma_red)
t0 = toto[4]
t1 = toto[int(len(toto) / 5) - 1]
t2 = toto[len(toto) - 1]
tmp = np.zeros_like(glock2)
tmp[t0 * a] = 1
tmp[t1 * a] = 1
tmp[t2 * a] = 1
tmp = tftft(tmp, wintype, win_duration, sigma, a, Fs2)


fig1, axs1 = plt.subplots(3)
axs1[0].plot(times_red, glock2[range(L_red * a)] / 10000, label="Glockenspiel")
axs1[0].grid()
axs1[0].legend()
axs1[0].xaxis.set_major_locator(ticker.NullLocator())
# axs1[1].plot(np.linspace(0, T2, len(sigma_red)), sigma_red,label='Focus')
axs1[1].plot(sigma_red, label="Focus")
axs1[1].grid()
axs1[1].legend(loc=1)
axs1[1].xaxis.set_major_locator(ticker.NullLocator())
axs1[2].plot(
    np.linspace(0, t_red_max, len(sigma_red)),
    np.real(tmp[0, : len(sigma_red)]),
    label="median/small/large",
)
axs1[2].grid()
axs1[2].legend()
axs1[2].set_xlabel("Time (sec.)")
plt.show()
