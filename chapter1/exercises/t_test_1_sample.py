# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 1. Parameter estimation and hypotesis testing.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, t

# Random seed
np.random.seed(42)

# One sample t-test
print("\nOne sample t-test:\nCompare sample mean to hypothesized value")

# Formulate hypotheses
# H0 (null hypothesis): The coin is fair (p = 0.5)
# H1 (alternative hypothesis): The coin is biased (p != 0.5)

# Simulate 100 coin tosses (1 = heads, 0 = tails)
# observations = np.random.choice([0, 1], size = 100, p = [0.5, 0.5])
observations = np.random.choice([0, 1], size = 100, p = [0.3, 0.7])

# Plot histogram of the observations
plt.hist(observations, bins = 2, edgecolor = "black", alpha = 0.75)
plt.xticks([0.25, 0.75], ["Tails (0)", "Heads (1)"])
plt.title("Histogram of 100 coin tosses")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.show()

# Compute observed mean and standard deviation
n = len(observations)
mean_observed = np.mean(observations)
std_dev_observed = np.std(observations, ddof = 1)  # Sample standard deviation

# Compute the t statistic and p-value Expected mean under H0 = 0.5
expected_mean = 0.5
t_stat_manual = (mean_observed - expected_mean) / (std_dev_observed / np.sqrt(n))
p_value_manual = 2 * (1 - t.cdf(abs(t_stat_manual), df = n-1))
print(f"\nT statistic (manual): {t_stat_manual:.4f}")
print(f"P-value (manual): {p_value_manual:.4f}")

# Compute the t statistic using a library
t_stat_scipy, p_value_scipy = ttest_1samp(observations, popmean = 0.5)
print(f"\nUsing SciPy library:")
print(f"\nT statistic (library): {t_stat_scipy:.4f}")
print(f"P-value (library): {p_value_scipy:.4f}")

# Parameters for plot t-distribution
x = np.linspace(-4, 4, 1000) # Range for t-distribution
t_dist = t.pdf(x, df = n - 1) # Probability density function for t-distribution

# Plot t-distribution
plt.plot(x, t_dist, label = f"t-distribution (df = {n - 1})")
plt.axvline(t_stat_manual, color = "red", linestyle = "--", label = f"T statistic = {t_stat_manual:.2f}")
plt.axvline(-t_stat_manual, color = "red", linestyle = "--")
plt.fill_between(x, t_dist, where = (x >= abs(t_stat_manual)), color = "red", alpha = 0.3, label = "Rejection region (one side)")
plt.fill_between(x, t_dist, where = (x <= -abs(t_stat_manual)), color = "red", alpha = 0.3)
plt.title("t-distribution and observed t statistic")
plt.xlabel("t value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Interpret result (significance level 0.05)
alpha = 0.05
if p_value_manual < alpha:
    print("Reject H0: The coin is likely biased.")
else:
    print("Accept H0: No evidence of biased coin.")
