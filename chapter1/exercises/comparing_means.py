# RCDS Introduction to probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 1. Parameter estimation and hypotesis testing.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Random seed
np.random.seed(0)

# Comparing means .............................................................
print("Comparing means")

# Generate two sets of gaussian observations
sample1 = np.random.normal(loc = 50, scale = 5, size = 30)
sample2 = np.random.normal(loc = 55, scale = 5, size = 30)

# Plot observations as histograms
plt.hist(sample1, bins = 15, alpha = 0.5, color = "blue", label = "Sample 1")
plt.hist(sample2, bins = 15, alpha = 0.5, color = "red", label = "Sample 2")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of gaussian observations")
plt.legend()
plt.show()

# Calculate t-statistic and p-value manually ..................................

# Calculate the sample means and standard deviations
mean1, mean2 = np.mean(sample1), np.mean(sample2)
std1, std2 = np.std(sample1, ddof = 1), np.std(sample2, ddof = 1)
n1, n2 = len(sample1), len(sample2)

# Calculate the standard error of the difference
se = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

# Calculate the t-statistic
t_stat_manual = (mean1 - mean2) / se

# Calculate degrees of freedom
df = n1 + n2 - 2

# Calculate two-tailed statistic p-value
p_value_manual = 2 * (1 - stats.t.cdf(abs(t_stat_manual), df = df))
print(f"Manual calculation:")
print(f"t-statistic = {t_stat_manual:.4f}")
print(f"p-value = {p_value_manual:.4f}")

# Calculate t-statistic and p-value with scipy ................................

# Calculate two-tailed statistic and p-value
t_stat_lib, p_value_lib = stats.ttest_ind(sample1, sample2)
print(f"\nUsing SciPy library:")
print(f"t-statistic = {t_stat_lib:.4f}")
print(f"p-value = {p_value_lib:.4f}")