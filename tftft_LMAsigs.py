import os
import sys
homedir = os.getenv("HOME")
sys.path.append(homedir + '/Science/git/focus/')

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile

from FocusFunctions import time_focus, time_focus_entropy, time_focus_renyi_entropy
from Transforms import tftft
from graphics import tfplot

# %% Global parameters

# Paths
refpath = homedir + '/Science/git/focus/'
respath = refpath + 'results/'
sigpath = refpath + 'signals/'

# Graphics
graph_file_flag = False
graph_file_ext = '.eps'

# %% Castanet SQAM signal
# load signal
fname = sigpath + 'casta.wav'
signame = 'casta'
Fs, casta = wavfile.read(fname)
casta = casta[:, 0]
T = len(casta)/Fs

# subsample
# subsampling = 8
# casta2 = scipy.signal.decimate(casta,subsampling,8)
# Fs2 = Fs/subsampling
# T2 = len(casta2)/Fs2
# times2 = np.linspace(0,T2,len(casta2))

# Shorten to 1 second
T_start = 0.1
start = int(T_start * Fs)
duration = 2 # seconds
T_end = T_start + duration
L = int(duration*Fs)
seg = range(start,start+L)
casta2 = casta[seg]
times2 = np.linspace(T_start,T_end,L)

# Choose reference window and length (corresponding to sigma=1)
wintype = "gauss"
# Time subsampling step (in time-frequency domain)
win_duration = 0.1 #(100 ms)
a = 16

# Compute focus function
sigma_max = 5
#sigma = time_focus(casta2, wintype, win_duration, a, Fs, sigma_max)
#sigma = time_focus_entropy(casta2, wintype, win_duration, a, Fs, sigma_max, 1)
p = 5
r = 0.01
sigma = time_focus_renyi_entropy(casta2,wintype,win_duration,a,Fs,sigma_max,p,r)
cst_focus = np.ones(sigma.shape)

fig, axs = plt.subplots(2)
axs[0].plot(times2, casta2)
axs[0].grid()
#axs[1].plot(np.linspace(0, T2, len(sigma)), sigma)
axs[1].plot(np.linspace(T_start, T_end, len(sigma)), sigma)
axs[1].grid()
if graph_file_flag == True:
    fname = respath + signame + '_tfoc' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Compute unfocused and focused transform
M_u = tftft(casta2, wintype, win_duration, cst_focus, a, Fs)
M_f = tftft(casta2, wintype, win_duration, sigma, a, Fs)

# Display unfocused transform
fig2 = plt.figure()
coef = tfplot(np.abs(M_u), a, np.array([0, 1]), fs=Fs, 
              display=False, dynrange=90)
_ = plt.imshow(coef, extent=(T_start, T_end, 0, Fs/2), interpolation='nearest', 
               aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_stft' + graph_file_ext
    fig2.savefig(fname)
plt.show()

# Display focused transform
fig3 = plt.figure()
coef = tfplot(np.abs(M_f), a, np.array([0, 1]), fs=Fs, 
              display=False, dynrange=90)
_ = plt.imshow(coef, extent=(T_start, T_end, 0, Fs/2), interpolation='nearest', 
               aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_tftft' + graph_file_ext
    fig3.savefig(fname)
plt.show()

# %% 
# =============================================================================
# Attaque forte
# =============================================================================
# load signal
fname = sigpath + 'Attaque-forte-sol.wav'
signame = 'attaque'
Fs, attaque = wavfile.read(fname)
seg = range(25000,70000)
attaque = attaque[seg]
T = len(attaque)/Fs
times = np.linspace(0,T,len(attaque))
# _ = plt.plot(times,attaque)
# plt.grid()

# subsample
# subsampling = 8
# attaque2 = scipy.signal.decimate(attaque,subsampling,8)
# Fs2 = Fs/subsampling
# T2 = len(attaque2)/Fs2
# times2 = np.linspace(0,T2,len(attaque2))


# Choose window
wintype = "gauss"
# Time subsampling step (in time-frequency domain)
win_duration = 0.1 #(100 ms)
a = 16

# Compute focus function
sigma_max = 5
p = 5
r = 0.01
#sigma = time_focus_entropy(attaque, wintype, win_duration, a, Fs, sigma_max, 1)
sigma = time_focus_renyi_entropy(attaque,wintype,win_duration,a,Fs,sigma_max,p,r)

fig, axs = plt.subplots(2)
axs[0].plot(times, attaque)
axs[0].grid()
axs[1].plot(np.linspace(0, T, len(sigma)), sigma)
axs[1].grid()
if graph_file_flag == True:
    fname = respath + signame + '_tfoc' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Compute unfocused and focused transform
cst_focus = np.ones(sigma.shape)
M_u = tftft(attaque, wintype, win_duration, cst_focus, a, Fs)
M_f = tftft(attaque, wintype, win_duration, sigma, a, Fs)

# Display unfocused transform
fig2 = plt.figure()
coef = tfplot(np.abs(M_u), a, np.array([0, 1]), fs=Fs, display=False, dynrange=90)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_stft' + graph_file_ext
    fig2.savefig(fname)
plt.show()

# Display focused transform
fig3 = plt.figure()
coef = tfplot(np.abs(M_f), a, np.array([0, 1]), fs=Fs, display=False, dynrange=90)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_tftft' + graph_file_ext
    fig3.savefig(fname)
plt.show()


#%%
# =============================================================================
# 
# =============================================================================
fname = sigpath + 'Attaque-douce-sol.wav'
signame = 'attaque'
Fs, attaque = wavfile.read(fname)
seg = range(35000,80000)
attaque = attaque[seg]
T = len(attaque)/Fs
times = np.linspace(0,T,len(attaque))
_ = plt.plot(times,attaque)
plt.grid()

# Choose window
wintype = "gauss"
# Time subsampling step (in time-frequency domain)
win_duration = 0.1 #(100 ms)
a = 16

# Compute focus function
sigma_max = 5
p = 2
r = 0.01
sigma = time_focus_entropy(attaque, wintype, win_duration, a, Fs, sigma_max, 1)
#sigma = time_focus_renyi_entropy(attaque,wintype,win_duration,a,Fs,sigma_max,p,r)

fig, axs = plt.subplots(2)
axs[0].plot(times, attaque)
axs[0].grid()
axs[1].plot(np.linspace(0, T, len(sigma)), sigma)
axs[1].grid()
if graph_file_flag == True:
    fname = respath + signame + '_tfoc' + graph_file_ext
    fig.savefig(fname)
plt.show()

# Compute unfocused and focused transform
cst_focus = np.ones(sigma.shape)
M_u = tftft(attaque, wintype, win_duration, cst_focus, a, Fs)
M_f = tftft(attaque, wintype, win_duration, sigma, a, Fs)

# Display unfocused transform
fig2 = plt.figure()
coef = tfplot(np.abs(M_u), a, np.array([0, 1]), fs=Fs, display=False, dynrange=90)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_stft' + graph_file_ext
    fig2.savefig(fname)
plt.show()

# Display focused transform
fig3 = plt.figure()
coef = tfplot(np.abs(M_f), a, np.array([0, 1]), fs=Fs, display=False, dynrange=90)
_ = plt.imshow(coef, extent=(0, T, 0, Fs/2), interpolation='nearest', aspect='auto',origin='lower')
_ = plt.colorbar()
_ = plt.xlabel('Time (sec.)')
_ = plt.ylabel('Frequency (Hz)')
if graph_file_flag:
    fname = respath + signame + '_tftft' + graph_file_ext
    fig3.savefig(fname)
plt.show()
