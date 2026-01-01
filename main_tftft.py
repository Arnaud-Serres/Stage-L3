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
import matplotlib.ticker as ticker
import scipy
from scipy.io import wavfile

from FocusFunctions import time_focus, time_focus_entropy, time_focus_renyi_entropy, time_focus_entropy_ref
from Transforms import tftft, itftft
from graphics import tfplot
from toy_signals import noisy_spikes_sine

# %% 
# =============================================================================
# Global parameters
# =============================================================================

# Paths
refpath = homedir + '/Science/git/focus/'
respath = refpath + 'results/'
sigpath = refpath + 'signals/'

# Spectrogram
alpha = 2

# Graphics
graph_file_flag = True
graph_file_ext = '.eps'

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
ampl = 1/20
times, x = noisy_spikes_sine(T, Fs, nb_spikes, psnr, f1, ampl)

# Window choice
wintype = "gauss"

# Time subsampling step (in time-frequency domain)
win_duration = 0.1 #(10 ms)
a = 4

# Parameters

# Focus function
sigma_max = 5
#sigma = time_focus(x, wintype, win_duration, a, Fs, sigma_max, normalized = False, n=1)
sigma = time_focus_entropy(x, wintype, win_duration, a, Fs, sigma_max, alpha)
cst_focus = np.ones(sigma.shape)
fig, axs = plt.subplots(2)
axs[0].plot(times, x)
axs[0].grid()
axs[1].plot(np.linspace(0, 1, len(sigma)), sigma)
axs[1].grid()
if graph_file_flag:
    fname = respath + signame + '_tfoc' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Unfocused and focused transforms
V_u = tftft(x, wintype, win_duration, cst_focus, a, Fs=Fs)
V_f = tftft(x, wintype, win_duration, sigma, a, Fs=Fs)

# Display transforms
fig2 = plt.figure()
coef = tfplot(np.abs(V_u), a, np.array([0, 1]), fs=Fs, display=False, 
              dynrange=dynrange)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', 
               aspect='auto', origin='lower', cmap=cmap)
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_stft' + graph_file_ext
    fig2.savefig(fname)
plt.show()

fig3 = plt.figure()
coef = tfplot(np.abs(V_f), a, np.array([0, 1]), fs=Fs, display=False, dynrange=60)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', 
               aspect='auto', origin='lower',cmap=cmap)
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_tftft' + graph_file_ext
    fig3.savefig(fname)
plt.show()

# Check inversion
x_rec = itftft(V_f, wintype, win_duration, sigma, len(x),Fs)
fig4, axs4 = plt.subplots(2)
axs4[0].plot(times,x,label="Original")
axs4[0].grid()
axs4[0].legend()
axs4[1].plot(times,x_rec,label="Reconstructed")
axs4[1].grid()
axs4[1].legend()
plt.show()

# %%
# Compute a focus function from the focused transform
sigma2 = time_focus_entropy_ref(V_f,sigma_max,2)
fig5,axs5 = plt.subplots(2)
axs5[0].plot(np.linspace(0, 1, len(sigma)), sigma,label="True")
axs5[0].grid()
axs5[0].legend()
axs5[1].plot(np.linspace(0, 1, len(sigma2)), sigma2,label="Approximate")
axs5[1].grid()
axs5[1].legend(loc = "lower left")
plt.show()

sigma_i = sigma2
nb_iter = 10
rec_err = np.zeros(nb_iter+1)
for it in range(nb_iter):
    # Use the estimated focus function within linear inverse transform
    x_rec_i =  itftft(V_f, wintype, win_duration, sigma_i, len(x),Fs)   #f = L dag M
    sigma_i = time_focus_entropy(x_rec_i,wintype,win_duration,a,Fs,sigma_max,2)
    rec_err[it] = np.linalg.norm(x_rec_i-x,ord=2)/np.linalg.norm(x,ord=2)

x_rec_end =  itftft(V_f, wintype, win_duration, sigma_i, len(x),Fs)
rec_err[nb_iter] = np.linalg.norm(x_rec_end-x,ord=2)/np.linalg.norm(x,ord=2)
plt.semilogy(rec_err,label="Reconstruction error")
plt.grid()
plt.xlabel("Iteration")
plt.legend()
plt.show()


# %% 
# =============================================================================
# Castanet SQAM signal
# =============================================================================

# load signal
fname = sigpath + 'casta.wav'
signame = 'casta'
Fs, casta = wavfile.read(fname)
casta = casta[:, 0]
T = len(casta)/Fs

# subsample
subsampling = 8
casta2 = scipy.signal.decimate(casta,subsampling,8)
Fs2 = Fs/subsampling
T2 = len(casta2)/Fs2
times2 = np.linspace(0,T2,len(casta2))

# Shorten
seg = range(5500,15500)
casta2 = casta2[seg]
times2 = times2[seg]
T2 = np.max(times2)

# Choose window
wintype = "gauss"
# Time subsampling step (in time-frequency domain)
win_duration = 0.1 #(10 ms)
a = 8

# Compute focus function
sigma_max = 5
#sigma = time_focus(casta2, wintype, win_duration, a, Fs2, sigma_max)
sigma = time_focus_entropy(casta2, wintype, win_duration, a, Fs2, sigma_max, 
                           alpha=alpha)
cst_focus = np.ones(sigma.shape)

fig, axs = plt.subplots(2)
axs[0].plot(times2, casta2)
axs[0].grid()
axs[1].plot(np.linspace(0, T2, len(sigma)), sigma)
axs[1].grid()
if graph_file_flag == True:
    fname = respath + signame + '_tfoc' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Display 3 sample windows
toto = np.argsort(-sigma)
t0 = toto[0]
t1 = toto[int(len(toto)/2)]
t2 = toto[len(toto)-1]
tmp = np.zeros_like(casta2)
tmp[t0*a] = 1
tmp[t1*a] = 1
tmp[t2*a] = 1
tmp = tftft(tmp, wintype, win_duration, sigma, a, Fs2)
#fig1, axs1 = plt.subplots(3)
fig1, axs1 = plt.subplots(2)
axs1[0].plot(times2, casta2/10000,label='Castanet')
axs1[0].grid()
axs1[0].legend()
axs1[0].xaxis.set_major_locator(ticker.NullLocator())
axs1[1].plot(np.linspace(0, T2, len(sigma)), sigma,label='Focus')
axs1[1].grid()
axs1[1].legend(loc=3)
axs1[1].xaxis.set_major_locator(ticker.NullLocator())
axs1[2].plot(np.linspace(0, T2, len(sigma)), np.real(tmp[0,:]),
              label='large/median/small')
axs1[2].grid()
axs1[2].legend()
axs1[2].set_xlabel("Time (sec.)")
plt.show()

# Compute unfocused and focused transform
V_u = tftft(casta2, wintype, win_duration, cst_focus, a, Fs2)
V_f = tftft(casta2, wintype, win_duration, sigma, a, Fs2)

# Display unfocused transform
fig2 = plt.figure()
coef = tfplot(np.abs(V_u), a, np.array([0, 1]), fs=Fs2, display=False, 
              dynrange=dynrange)
_ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest', 
               aspect='auto',origin='lower',cmap=cmap)
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_stft' + graph_file_ext
    fig2.savefig(fname)
plt.show()

# Display focused transform
fig3 = plt.figure()
coef = tfplot(np.abs(V_f), a, np.array([0, 1]), fs=Fs2, display=False, 
              dynrange=dynrange)
_ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest', 
               aspect='auto',origin='lower',cmap=cmap)
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_tftft' + graph_file_ext
    fig3.savefig(fname)
plt.show()

# %% 

# # =============================================================================
# # Glockenspiel signal
# # =============================================================================

# # load signal
# fname = sigpath + 'glock.wav'
# signame = 'glock'
# Fs, glock = wavfile.read(fname)
# T = len(glock)/Fs
# times = np.linspace(0,T,len(glock))

# # Subsample
# subsampling = 4
# glock2 = scipy.signal.decimate(glock,subsampling,8)
# Fs2 = Fs/subsampling

# # shorten
# seg = range(40000)
# glock2 = glock2[seg]
# T2 = len(glock2)/Fs2
# times2 = np.linspace(0,T2,len(glock2))

# # Window choice
# wintype = "gauss"
# # Time subsampling step (in time-frequency domain)
# win_duration = 0.2 #(10 ms)
# a = 8

# # Max focus
# sigma_max = 5

# # ## STANDARD FOCUSED FUNCTION
# # # Compute standard focus function
# # sigma = time_focus(glock2, wintype, win_duration, a, Fs2, sigma_max, False,
# #                    alpha = alpha, n = 2)
# # cst_focus = np.ones(sigma.shape)
# # fig, axs = plt.subplots(2)
# # axs[0].plot(times2, glock2)
# # axs[0].grid()
# # axs[1].plot(np.linspace(0,T2,len(sigma)),sigma)
# # axs[1].grid()
# # if graph_file_flag == True:
# #     fname = respath + signame + '_tfoc' + graph_file_ext
# #     fig.savefig(fname)
# # plt.show()

# # # Compute unfocused and focused transform
# # V_u = tftft(glock2, wintype, win_duration, cst_focus, a, Fs2)
# # V_f = tftft(glock2, wintype, win_duration, sigma, a, Fs2)

# # # Display unfocused transform
# # fig2 = plt.figure()
# # coef = tfplot(np.abs(V_u), a, np.array([0, 1]), fs=Fs2, display=False, 
# #               dynrange=dynrange)
# # _ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest', 
# #                aspect='auto', origin='lower',  cmap=cmap)
# # _ = plt.colorbar()
# # _ = plt.xlabel('Time (sec.)')
# # _ = plt.ylabel('Frequency (Hz)')
# # if graph_file_flag:
# #     fname = respath + signame + '_stft' + graph_file_ext
# #     fig2.savefig(fname)
# # plt.show()

# # # Display focused transform
# # fig3 = plt.figure()
# # coef = tfplot(np.abs(V_f), a, np.array([0, 1]), fs=Fs2, display=False, 
# #               dynrange=dynrange)
# # _ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest', 
# #                aspect='auto', origin='lower', cmap = cmap)
# # _ = plt.colorbar()
# # _ = plt.xlabel('Time (sec.)')
# # _ = plt.ylabel('Frequency (Hz)')
# # if graph_file_flag:
# #     fname = respath + signame + '_tftft' + graph_file_ext
# #     fig3.savefig(fname)
# # plt.show()

# ## ENTROPY-BASED FOCUSED FUNCTION
# # Compute standard focus function
# #sigma = time_focus_entropy(glock2, wintype, win_duration, a, Fs2, sigma_max, alpha=alpha)
# r = 0.00001
# sigma = time_focus_renyi_entropy(glock2,wintype,win_duration,a,Fs2,sigma_max,4,0.01)
# cst_focus = np.ones(sigma.shape)
# fig, axs = plt.subplots(2)
# axs[0].plot(times2, glock2)
# axs[0].grid()
# axs[1].plot(np.linspace(0,T2,len(sigma)),sigma)
# axs[1].grid()
# if graph_file_flag == True:
#     fname = respath + signame + '_tfoc_ent' + graph_file_ext
#     fig.savefig(fname)
# plt.show()

# # Compute unfocused and focused transform
# V_u = tftft(glock2, wintype, win_duration, cst_focus, a, Fs2)
# V_f = tftft(glock2, wintype, win_duration, sigma, a, Fs2)

# # Display unfocused transform
# fig2 = plt.figure()
# coef = tfplot(np.abs(V_u), a, np.array([0, 1]), fs=Fs2, display=False, 
#               dynrange = dynrange)
# _ = plt.imshow(coef, extent=(0, T2, 0, Fs2/2), interpolation='nearest', 
#                aspect='auto', origin='lower', cmap = cmap)
# _ = plt.colorbar()
# _ = plt.xlabel('Time (sec.)')
# _ = plt.ylabel('Frequency (Hz)')
# if graph_file_flag:
#     fname = respath + signame + '_stft_ent' + graph_file_ext
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
#     fname = respath + signame + '_tftft_ent' + graph_file_ext
#     fig3.savefig(fname)
# plt.show()


