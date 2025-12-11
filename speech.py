#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:54:44 2025

@author: bruno
"""

import os
import sys
homedir = os.getenv("HOME")
sys.path.append(homedir + '/Science/git/focus/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Transforms import tftft
from graphics import tfplot

# %% Global parameters

# Paths
refpath = homedir + '/Science/git/focus/'
respath = refpath + 'results/'
sigpath = refpath + 'signals/'

# Graphics
graph_file_flag = False
graph_file_ext = '.png'


# %%
# =============================================================================
# Read signals
# =============================================================================
fname = sigpath + 'heed_m.wav'
signame = 'heed_m'
Fs, sig = wavfile.read(fname)
L = len(sig)
T = L/Fs
times = np.linspace(0,T,len(sig))
plt.plot(times,sig)
plt.xlabel('Time (sec.)')
plt.title(signame)
plt.grid()

# %%
# STFT
# Choose window
wintype = "gauss"
# Time subsampling step (in time-frequency domain)
win_duration = 0.075 #(10 ms)
a = 1
N = int(L/a)

cst_focus = np.ones(N)
V = tftft(sig, wintype, win_duration, cst_focus, a, Fs)
M = V.shape[0]
M2 = int(M/2)
V2 = V[range(M2),:]

fig2 = plt.figure()
coef = tfplot(np.abs(V2), a, np.array([0, 1]), fs=Fs, display=False, dynrange=60)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/4), interpolation='nearest', aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
plt.title(signame)
if graph_file_flag:
    fname = respath + signame + '_stft' + graph_file_ext
    fig2.savefig(fname)
plt.show()