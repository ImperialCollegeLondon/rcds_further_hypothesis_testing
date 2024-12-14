# RCDS Introduction to probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 1. Parameter estimation and hypotesis testing.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, t

# Random seed
np.random.seed(0)

# Two sample t-test
print("\nTwo sample t-test:\nCompare sample means of two independent groups")

# Formulate hypotheses
# H0: x1, x2 come from sampe distribution (mean1 = mean2 = mu)
# H1: x1, x2, come from different distriubtions (mean1 != mean2 != mu)

# Simulate two sets of gaussian observations
sample1 = np.random.normal(loc = 50, scale = 5, size = 30)
sample2 = np.random.normal(loc = 55, scale = 5, size = 30)

# Plot observations as histograms
plt.hist(sample1, bins = 15, alpha = 0.5, color = "blue", label = "Sample 1")
plt.hist(sample2, bins = 15, alpha = 0.5, color = "red", label = "Sample 2")
plt.title("Histogram of gaussian observations")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Compute the sample means and standard deviations
mean1, mean2 = np.mean(sample1), np.mean(sample2)
std1, std2 = np.std(sample1, ddof = 1), np.std(sample2, ddof = 1)
n1, n2 = len(sample1), len(sample2)
# Compute standard error of the difference
se = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

# Compute the t-statistic
t_stat_manual = (mean1 - mean2) / se
p_value_manual = 2 * (1 - t.cdf(abs(t_stat_manual), df = n1 + n2 - 2))
print(f"Manual calculation:")
print(f"t-statistic = {t_stat_manual:.4f}")
print(f"p-value = {p_value_manual:.4f}")

# Compute two-tailed statistic and p-value
t_stat_scipy, p_value_scipy = ttest_ind(sample1, sample2)
print(f"\nUsing SciPy library:")
print(f"t-statistic = {t_stat_scipy:.4f}")
print(f"p-value = {p_value_scipy:.4f}")

# Parameters for plot t-distribution
x = np.linspace(-4, 4, 1000) # Range for t-distribution
t_dist = t.pdf(x, df = n2 + n1 - 2) # Probability density function for t-distribution

# Plot t-distribution
plt.plot(x, t_dist, label = f"t-distribution (df = {n2 + n1 -1})")
plt.axvline(t_stat_manual, color = "red", linestyle = "--", label = f"T statistic = {t_stat_manual:.2f}")
plt.axvline(-t_stat_manual, color = "red", linestyle = "--")
plt.fill_between(x, t_dist, where = (x >= abs(t_stat_manual)), color = "red", alpha = 0.3, label = "Rejection Region (one side)")
plt.fill_between(x, t_dist, where = (x <= -abs(t_stat_manual)), color = "red", alpha = 0.3)
plt.title("t-distribution and observed t statistic")
plt.xlabel("t value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Interpret result (significance level 0.05)
alpha = 0.05
if p_value_manual < alpha:
    print("Reject H0: Samples come from different distributions.")
else:
    print("Accept H0: Samples come from same distribution.")
