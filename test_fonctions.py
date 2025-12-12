from fonction_focus import A_to_max_focus, max_focus_to_A, rmin
import math
from toy_signals import noisy_spikes_sine
from FocusFunctions import make_stftwin
import numpy as np
import matplotlib.pyplot as plt

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
# sigma_max = 10
sigma_min = 0.1
A = 100
u = np.zeros(x.shape[0])
u[0] = 1

sigma = np.ones(x.shape[0])
winlen = math.floor(0.1 * 8000)
h = make_stftwin(0, "gauss", winlen, x.shape[0], sigma)
r = rmin(h, sigma_min, u, alpha, A)
p = 4
nb_iter = 40

print(max_focus_to_A(A_to_max_focus(A, h, alpha, r), alpha, h, sigma_min, u))

nb_points = 50
L = [1]
for i in range(1, nb_iter):
    r = rmin(h, sigma_min, u, p, i)
    L.append(A_to_max_focus(i, h, alpha, r))

plt.plot(L, label="Max_focus function")
plt.grid()
plt.xlabel("A")
plt.legend()
plt.show()
