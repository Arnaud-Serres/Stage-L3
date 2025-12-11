#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:53:35 2025

@author: bruno
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from scipy.signal.windows import hann, gaussian
from scipy.signal import chirp
import numpy as np
import matplotlib.pyplot as plt

from Transforms import tftft
from graphics import tfplot

# %%

# =============================================================================
# Gabor Atoms Examples
# =============================================================================

L = 1024
times = np.linspace(-1/2,1/2,L)

# Gaussian window
win0 = gaussian(L,32)
plt.plot(times,win0,linewidth=1,color='b')
win = np.roll(win0 * np.cos(2*np.pi*16*np.linspace(0,1,L)),-256)
plt.plot(times,win,linewidth=1,color='b')
win = np.roll(win0 * np.sin(2*np.pi*16*np.linspace(0,1,L)),-256)
plt.plot(times,win,linewidth=1.5,linestyle='--',color='r')
win = np.roll(win0 * np.cos(2*np.pi*32*np.linspace(0,1,L)),350)
plt.plot(times,win,linewidth=1,color='b')
win = np.roll(win0 * np.sin(2*np.pi*32*np.linspace(0,1,L)),350)
plt.plot(times,win,linewidth=1.5,linestyle='--',color='r')
plt.grid()
plt.xlabel('Time (sec.)')
plt.title('Gaussian Gabor atoms')
plt.show()

# Hann window
tmp = hann(int(L/6))
win_len = len(tmp)
win0 = np.zeros(L)
L_over_2 = int(L/2)
win_start = L_over_2 - int(win_len/2)
seg = range(win_start,win_start+win_len)
win0[seg] = tmp
plt.plot(times,win0,linewidth=1,color='b')
win = np.roll(win0 * np.cos(2*np.pi*16*np.linspace(0,1,L)),-256)
plt.plot(times,win,linewidth=1,color='b')
win = np.roll(win0 * np.sin(2*np.pi*16*np.linspace(0,1,L)),-256)
plt.plot(times,win,linewidth=1.5,linestyle='--',color='r')
win = np.roll(win0 * np.cos(2*np.pi*32*np.linspace(0,1,L)),350)
plt.plot(times,win,linewidth=1,color='b')
win = np.roll(win0 * np.sin(2*np.pi*32*np.linspace(0,1,L)),350)
plt.plot(times,win,linewidth=1.5,linestyle='--',color='r')
plt.grid()
plt.xlabel('Time (sec.)')
plt.title('Hann Gabor atoms')
plt.show()


# %%
# =============================================================================
# Play with chirps
# =============================================================================

T = 4
Fs = 1024
L = int(T*Fs)
t = np.linspace(0,T,L)
sig = chirp(t,50,T,450,method="hyperbolic")
plt.plot(t,sig)
cst_focus = np.ones_like(t)
wintype = "gauss"
# Time subsampling step (in time-frequency domain)
win_duration = 0.2 #(10 ms)
a = 1
N = int(L/a)

cst_focus = np.ones(N)
V = tftft(sig, wintype, win_duration, cst_focus, a, Fs)
fig = plt.figure()
coef = tfplot(np.abs(V), a, np.array([0, 1]), fs=Fs, display=False, dynrange=60)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
plt.title("chirp")


# %%
