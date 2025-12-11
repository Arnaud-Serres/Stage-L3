#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 18:56:07 2025

@author: bruno
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from scipy.signal.windows import gaussian
from scipy.signal import chirp
import numpy as np
import matplotlib.pyplot as plt

from Transforms import  cwt, sqcauchy_wavelet
from graphics import tsplot

# %%
# =============================================================================
# Wavelet examples
# =============================================================================

# Mexican hat (real)
L = 1024
cst = 2*np.sqrt(2)
times = np.linspace(-1/2,1/2,L)
wavelet = - np.diff(np.diff(gaussian(L+2,32)))*cst
plt.plot(times,np.roll(wavelet,-300),color='b')
wavelet = - np.diff(np.diff(gaussian(L+2,16)))
plt.plot(times,wavelet,color='b')
wavelet = - np.diff(np.diff(gaussian(L+2,8)))/cst
plt.plot(times,np.roll(wavelet,256),color='b')
plt.grid()
plt.title("Mexican hat wavelets'")
plt.xlabel('Time (sec.)')
plt.show()

# Cauchy-type wavelets
L_over_2 = int(L/2)
scale = 8
n = 4
wavelet = sqcauchy_wavelet(L,n,scale)
plt.plot(times, np.roll(np.real(wavelet),L_over_2),color='b')
plt.plot(times, np.roll(np.imag(wavelet),L_over_2),color='r',linestyle='--')
scale = 16
wavelet = sqcauchy_wavelet(L,n,scale)
plt.plot(times, np.roll(np.real(wavelet),L_over_2 - 256),color='b')
plt.plot(times, np.roll(np.imag(wavelet),L_over_2 - 256),color='r',linestyle='--')
scale = 6
wavelet = sqcauchy_wavelet(L,n,scale)
plt.plot(times, np.roll(np.real(wavelet),L_over_2 + 256),color='b')
plt.plot(times, np.roll(np.imag(wavelet),L_over_2 + 256),color='r',linestyle='--')
plt.grid()
plt.title("Cauchy-type wavelets")
plt.xlabel('Time (sec.)')
plt.show()


# %%
# =============================================================================
# Play with singularities
# =============================================================================
T = 3;
Fs = 1024;
L = int(T*Fs)
t = np.linspace(0,T,L)
sig = np.zeros(L)
sig[Fs] = 1
sig[2*Fs] = 1
sig[2*Fs+1] = -1
plt.plot(sig)
plt.xlabel('Time (sec.)')
plt.grid()
plt.show()


min_scale = 0
max_scale = 6
nb_scales = 100
scales = np.linspace(min_scale, max_scale, nb_scales)
scales = 2**(scales)

W = cwt(sig,4,scales,Fs=Fs)

fig = plt.figure()
W2 = np.diag(1/np.sqrt(scales)) @ W
coef = tsplot(np.abs(W2), 1/Fs, np.array((max_scale, min_scale)), dynrange=60)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto', origin='upper')
#_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
plt.show()

fig2 = plt.Figure()
plt.semilogy(W2[:,Fs])
plt.semilogy(W2[:,2*Fs])
plt.xlabel('log(Scale)')
plt.ylabel('Fixed time transform magnitude')
plt.legend(('t=1','t=2'))
plt.grid()
plt.show()

# %%
# =============================================================================
# Play with chirps
# =============================================================================

T = 4
Fs = 1024
L = int(T*Fs)
t = np.linspace(0,T,L)
sig = chirp(t,10,T,450,method="hyperbolic")
plt.plot(t,sig)

min_scale = -1
max_scale = 6
nb_scales = 100
scales = np.linspace(min_scale, max_scale, nb_scales)
scales = 2**(scales)

W = cwt(sig,8,scales,Fs=Fs)

fig3 = plt.figure()
W2 = np.diag(1/np.sqrt(scales)) @ W
coef = tsplot(np.abs(W2), 1/Fs, np.array((max_scale, min_scale)), dynrange=40)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto', origin='upper')
#_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
plt.show()