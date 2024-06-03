# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Bayesian Inference: Updating beliefs about a coin's bias
def bayesian_inference(data, prior_params, likelihood_params):
    prior_alpha, prior_beta = prior_params
    heads, tails = likelihood_params
    posterior_alpha = prior_alpha + heads
    posterior_beta = prior_beta + tails
    return posterior_alpha, posterior_beta

# Bayesian Modeling: Gaussian data fitting with known variance
def bayesian_modeling(data, prior_params):
    prior_mean, prior_std = prior_params
    likelihood_mean = np.mean(data)
    likelihood_std = np.std(data)
    posterior_mean = (prior_mean / prior_std**2 + np.sum(data) / likelihood_std**2) / (1 / prior_std**2 + len(data) / likelihood_std**2)
    posterior_std = np.sqrt(1 / (1 / prior_std**2 + len(data) / likelihood_std**2))
    return posterior_mean, posterior_std

# Bayesian Computation: Sampling from a posterior distribution using MCMC (Metropolis-Hastings)
def metropolis_hastings(data, num_samples):
    samples = [0]  # initial value
    for _ in range(num_samples):
        candidate = np.random.normal(samples[-1], 1)  # proposal distribution
        likelihood_current = np.prod(norm.pdf(data, loc=samples[-1]))
        likelihood_candidate = np.prod(norm.pdf(data, loc=candidate))
        acceptance_ratio = likelihood_candidate / likelihood_current
        if acceptance_ratio >= 1 or np.random.uniform(0, 1) < acceptance_ratio:
            samples.append(candidate)
        else:
            samples.append(samples[-1])
    return samples

# Example data
coin_flips = np.random.choice([0, 1], size=100, p=[0.3, 0.7])  # 1 for heads, 0 for tails
gaussian_data = np.random.normal(loc=5, scale=2, size=100)

# Bayesian Inference example
prior_params = (1, 1)  # Beta(1, 1) prior
likelihood_params = (np.sum(coin_flips), len(coin_flips) - np.sum(coin_flips))
posterior_alpha, posterior_beta = bayesian_inference(coin_flips, prior_params, likelihood_params)
print("Bayesian Inference: Posterior parameters (alpha, beta):", posterior_alpha, posterior_beta)

# Bayesian Modeling example
prior_params = (0, 1)  # Gaussian prior with mean=0, std=1
posterior_mean, posterior_std = bayesian_modeling(gaussian_data, prior_params)
print("Bayesian Modeling: Posterior mean and std:", posterior_mean, posterior_std)

# Bayesian Computation example
samples = metropolis_hastings(gaussian_data, num_samples=1000)
plt.hist(samples, bins=30, density=True, alpha=0.5, label="Samples")
x = np.linspace(np.min(samples), np.max(samples), 100)
plt.plot(x, norm.pdf(x, loc=posterior_mean, scale=posterior_std), "r-", lw=2, label="True Posterior")
plt.title("Bayesian Computation: MCMC Sampling")
plt.xlabel("Parameter Value")
plt.ylabel("Density")
plt.legend()
plt.show()