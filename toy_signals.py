import numpy as np

# %%

def noisy_spikes_sine(T,Fs, nb_spikes, psnr, f1, ampl):
    L = T * Fs
    times = np.linspace(0, T, L)

    # Random spikes at random locations
    spike_loc = np.random.permutation(L)[range(nb_spikes)]
    nsp = np.zeros(L)
    nsp[spike_loc] = np.random.normal(0, 1, nb_spikes)
    x = nsp

    # Add noise
    x += np.random.normal(0, 1 / psnr, L)

    # Add sine wave
    x += ampl * np.sin(2 * np.pi * times * f1)

    return times, x

