# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - Imperial College London
# Chapter 1 - One sample t-test

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Formulate null and alternative hypothesis
# H0: die is fair, p = 1/2
# H1: die is biased, p != 1/2
alpha = 0.05 # significance level

# Example data
die_rolls = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

# Plot to visualize data
plt.figure(figsize = (8, 6))
plt.hist(die_rolls, bins = range(1, 8), edgecolor = "black", align = "left", rwidth = 0.8)
plt.xlabel("die roll")
plt.ylabel("frequency")
plt.title("Histogram of die rolls")
plt.xticks(range(1, 7))
plt.grid(axis = "y", alpha = 0.75)
plt.show()

# One sample t-test: manual
n = len(die_rolls)
expected_mean = 3.5
sample_mean = np.mean(die_rolls)
sample_std = np.std(die_rolls, ddof = 1)  # ddof = 1 for sample standard deviation
print("\nSample size: ", n, "\nobserved mean: ", sample_mean, "\novserved std: ", sample_std)

# One sample t-test: manual
t_statistic_manual = (sample_mean - expected_mean) / (sample_std / np.sqrt(n))
p_value_manual = 2 * (1 - stats.t.cdf(np.abs(t_statistic_manual), df = n-1))
print(f"\nt-statistic - manual = {t_statistic_manual}, p-value = {p_value_manual}")

# One sample t-test: scipy
t_statistic_scipy, p_value_scipy = stats.ttest_1samp(die_rolls, expected_mean)
print(f"t-statistic - scipy = {t_statistic_scipy}, p-value = {p_value_scipy}")