# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - Imperial College London
# Chapter 1 - Chi2 test

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulate two distributions
np.random.seed(123) 
sample1 = np.random.normal(50, 5, 50)
sample2 = np.random.normal(62, 5, 50)

# Create bins for the histograms
bins = np.histogram_bin_edges(np.concatenate((sample1, sample2)), bins = "auto")

# Plot to visualize data
plt.hist(sample1, bins = bins, alpha = 0.5, label = "Sample 1")
plt.hist(sample2, bins = bins, alpha = 0.5, label = "Sample 2")
plt.xlabel("value")
plt.ylabel("frequency")
plt.legend(loc = "upper right")
plt.title("Histograms of Sample 1 and Sample 2")
plt.grid(axis = "y", alpha = 0.75)
plt.legend()
plt.show()

# Compute the observed and expected frequencies
obs_freq, _ = np.histogram(sample1, bins = bins)
exp_freq, _ = np.histogram(sample2, bins = bins)

# Ensure no expected frequency is zero to avoid division by zero
exp_freq = np.where(exp_freq == 0, 1e-10, exp_freq)

# Manual chi-squared test calculation
chi2_stat_manual = np.sum((obs_freq - exp_freq)**2 / exp_freq)
df = len(obs_freq) - 1
p_value_manual = 1 - stats.chi2.cdf(chi2_stat_manual, df)
print(f"Manual chi-squared test:\n Chi-squared statistic: {chi2_stat_manual:.4f}, p-value: {p_value_manual:.4f}")

# Using scipy.stats.chisquare
chi2_stat_scipy, p_value_scipy = stats.chisquare(obs_freq, f_exp=exp_freq)
print(f"\nscipy.stats.chisquare:\n Chi-squared statistic: {chi2_stat_scipy:.4f}, p-value: {p_value_scipy:.4f}")