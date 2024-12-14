# RCDS Introduction to probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 1. Parameter estimation and hypotesis testing.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, levene, bartlett, f

# Random seed
np.random.seed(42)

# Fisher test
print("\nFisher test:\nCompare sample variances of two independent groups")

# Simulate two sets of gaussian observations
sample1 = np.random.normal(loc = 50, scale = 5, size = 30)
sample2 = np.random.normal(loc = 55, scale = 7, size = 30)

# Plot observations as histograms
plt.hist(sample1, bins = 15, alpha = 0.5, color = "blue", label = "Sample 1")
plt.hist(sample2, bins = 15, alpha = 0.5, color = "red", label = "Sample 2")
plt.title("Histogram of gaussian Observations")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Calculate variances of the two samples
var1, var2 = np.var(sample1, ddof = 1), np.var(sample2, ddof = 1)
# Degrees of freedom for each sample
df1 = len(sample1) - 1
df2 = len(sample2) - 1

# Calculate F-statistic and p-value manually
F_stat_manual = var1 / var2 if var1 > var2 else var2 / var1
p_value_manual = 2 * (1 - f.cdf(F_stat_manual, df1, df2))
print(f"Manual calculation:")
print(f"F-statistic = {F_stat_manual:.4f}")
print(f"p-value = {p_value_manual:.4f}")

# Calculate F statistic and p-value using library
F_stat_scipy, p_value_scipy = f_oneway(sample1, sample2)
# F_stat_scipy, p_value_scipy = levene(sample1, sample2)
# F_stat_scipy, p_value_scipy = bartlett(sample1, sample2)
print(f"\nUsing SciPy library:")
print(f"F-statistic = {F_stat_scipy:.4f}")
print(f"p-value = {p_value_scipy:.4f}")

# Plot the F-distribution
x = np.linspace(0, 5, 1000)  # Range of F-values for plotting
f_dist = f.pdf(x, df1, df2)  # F-distribution density function

# Plot F-distribution
plt.plot(x, f_dist, label = f"F-distribution (df1 = {df1}, df2 = {df2})")
plt.axvline(F_stat_manual, color = "red", linestyle = "--", label = f"Observed F = {F_stat_manual:.2f}")
plt.fill_between(x, f_dist, where = (x >= F_stat_manual), color = "red", alpha = 0.3, label = "Rejection region")
plt.title("F-distribution and observed F-statistic")
plt.xlabel("F value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Interpret result (significance level 0.05)
alpha = 0.05
if p_value_manual < alpha:
    print("Reject H0: Samples come from different distribution.")
else:
    print("Accept H0: Samples come from same distribution.")
