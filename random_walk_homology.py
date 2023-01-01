import os, math, random

import numpy as np

from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding
from gtda.time_series import TakensEmbedding

import matplotlib.pyplot as plt

def brownian_motion(N, T, h, seed=42):
    """
    Simulates Brownian motion
    :param int N : the number of discrete steps
    :param int T: the number of continuous time steps
    :param float h: the variance of the increments
    :param int seed: initial seed of the random generator
    """
    np.random.seed(seed)
    dt = 1. * T/N
    random_increments = np.random.normal(0.0, 1.0 * h, N)*np.sqrt(dt)
    brownian_motion = np.cumsum(random_increments)
    brownian_motion = np.insert(brownian_motion, 0, 0.0)

    return brownian_motion, random_increments

def drifted_brownian_motion(mu, sigma, N, T, seed=42):
    """Simulates Brownian Motion with drift
    :param float mu: drift coefficient
    :param float sigma: volatility coefficient
    :param int N : number of discrete steps
    :param int T: number of continuous time steps
    :param int seed: initial seed of the random generator
    """
    np.random.seed(seed)
    W, _ = brownian_motion(N, T, 1.0)
    dt = 1. * T/N
    time_steps = np.linspace(0.0, N*dt, N+1)
    X = mu * time_steps + sigma * W

    return X

# Random walk params
N = 10000
T = 1000
h = 1 
seed = 2022

# TDA Params
VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
STE = SingleTakensEmbedding(parameters_type='search', dimension=2, time_delay=50, n_jobs=-1)

X, epsilon = brownian_motion(N, T, h, seed)
X_td = STE.fit_transform(X)

fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]}, figsize=(20,25), dpi=300)

a0.plot(X, label="$\mu = 0,\, \sigma = 1$")
a1.scatter(X_td[:,0], X_td[:,1], s=0.4, alpha=0.8, label="$\mu = 0,\, \sigma = 1$")

# Param Sweep
mu = [0, 0, 0.05, 0.1]
sigma = [0.5, 1.8, 1, 1]

for i in range(len(mu)):
    
    X = drifted_brownian_motion(mu[i], sigma[i], N, T, seed)
    X_td = STE.fit_transform(X)
    a0.plot(X, label="$\mu = " + str(mu[i]) + ",\, \sigma = " + str(sigma[i]) + "$")
    a1.scatter(X_td[:,0], X_td[:,1], s=0.4, alpha=0.8,
               label="$\mu = " + str(mu[i]) + ",\, \sigma = " + str(sigma[i]) + "$")

a0.set_ylabel("position", fontsize=15)
a0.set_xlabel("time steps", fontsize=15)
a0.set_title("$X_t = \mu t + \sigma W_t$ (Random Seed: " + str(seed) + ")", fontsize=18)
a0.legend(frameon=False, fontsize=14)

a1.set_ylabel("$t$", fontsize=25)
a1.set_xlabel("$t - \delta t$", fontsize=25)
a1.set_title("$X_t = \mu t + \sigma W_t$ (Random Seed: " + str(seed) + ")", fontsize=18)

lgnd = a1.legend(frameon=False, fontsize=18)
for i in range(len(mu)+1):
    lgnd.legendHandles[i]._sizes = [30]

fig.tight_layout()
fig.savefig("Brownian_Motion_1D.png")
plt.show()
