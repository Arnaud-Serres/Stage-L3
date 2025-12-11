#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2025

@author: bruno
"""

import os
import sys
homedir = os.getenv("HOME")
sys.path.append(homedir + '/Science/git/focus/')

import numpy as np
import matplotlib.pyplot as plt
import scipy

from FocusFunctions import frequency_focus_entropy
from Transforms import fftft
from graphics import tsplot
from toy_signals import noisy_spikes_sine

# %% Global parameters

# Paths
refpath = homedir + '/Science/git/focus/'
respath = refpath + 'results/'
sigpath = refpath + 'signals/'

# Graphics
graph_file_flag = True
graph_file_ext = '.eps'

# %% Random spikes + sines + noise
# Generate toy signal composed of randomly located spikes with random amplitudes, sine waves and additive noise
signame = "noisy_spikesin"

# Set signal parameters
T = 1
Fs = 8000
L = T * Fs
times = np.linspace(0, T, L)

# Generate signal
nb_spikes = 50
spike_loc = np.random.permutation(L)[range(nb_spikes)]
x = np.zeros(L)
x[spike_loc] = np.random.normal(0, 1, nb_spikes)
# add noise
psnr = 25
x += np.random.normal(0, 1/psnr, L)
# add sine waves
f1 = 10
ampl = 1/10
x += ampl * np.cos(2*np.pi*times*f1)
x += ampl * np.cos(2*np.pi*times*2*f1)
x += ampl * np.cos(2*np.pi*times*4*f1)
x += ampl * np.cos(2*np.pi*times*64*f1)
# Periodogram
fr, x_pgram = scipy.signal.periodogram(x, fs=Fs)
# Display
fig, axs = plt.subplots(2)
_ = axs[0].plot(times, x)
_ = axs[0].grid()
_ = axs[1].loglog(fr[1:], x_pgram[1:])
_ = axs[1].grid()
_ = axs[1].set_xlabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_sigspec' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Transform parameters
n = 4
min_scale = 0
max_scale = 9
nb_scales = 200
scales = np.linspace(min_scale, max_scale, nb_scales)
scales = 2**(-scales)

# Unfocused transform
cst_focus = np.ones(len(scales))
W_u = fftft(x, n, scales, cst_focus)
W_spectrum = np.mean(np.abs(W_u), axis=1)

# Focus function parameters
sigma_ref = 1
sigma_max = 5

# Focus function
sigma = frequency_focus_entropy(x,n,scales,sigma_ref,sigma_max,alpha=1)

# Display
fig2, axs2 = plt.subplots(2)
_ = axs2[0].plot(W_spectrum)
_ = axs2[0].grid()
_ = axs2[1].plot(sigma)
_ = axs2[1].grid()
_ = axs2[1].set_xlabel('scale index')
if graph_file_flag:
    fname = respath + signame + '_ffoc' + graph_file_ext
    fig2.savefig(fname)
plt.show()


# Focused transform
W_f = fftft(x, n, scales, sigma)

# Display
fig3 = plt.figure()
coef = tsplot(np.abs(W_u), 1/Fs, np.array((max_scale, min_scale)), dynrange=60)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto', origin='upper')
#_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_cwt' + graph_file_ext
    fig3.savefig(fname)
plt.show()

fig4 = plt.figure()
coef = tsplot(np.abs(W_f), 1/Fs, np.array((max_scale, min_scale)), dynrange=60)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto', origin='upper')
#_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_fftft' + graph_file_ext
    fig4.savefig(fname)
plt.show()