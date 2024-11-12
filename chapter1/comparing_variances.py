# RCDS Introduction to probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 1. Parameter estimation and hypotesis testing.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Random seed
np.random.seed(0)

# Comparing variances .........................................................
print("Comparing variances")

# Generate two sets of gaussian observations
sample1 = np.random.normal(loc = 50, scale = 5, size = 30)
sample2 = np.random.normal(loc = 55, scale = 7, size = 30)

# Plot observations as histograms
plt.hist(sample1, bins = 15, alpha = 0.5, color = "blue", label = "Sample 1")
plt.hist(sample2, bins = 15, alpha = 0.5, color = "red", label = "Sample 2")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of gaussian observations")
plt.legend()
plt.show()

# Perform F-test manually .....................................................

# Calculate variances of the two samples
var1, var2 = np.var(sample1, ddof = 1), np.var(sample2, ddof = 1)

# Calculate F-statistic (larger variance / smaller variance)
F_stat_manual = var1 / var2 if var1 > var2 else var2 / var1

# Degrees of freedom for each sample
df1 = len(sample1) - 1
df2 = len(sample2) - 1

# Calculate p-value manually using the F-distribution
p_value_manual = 2 * (1 - stats.f.cdf(F_stat_manual, df1, df2))

print(f"Manual F-test:")
print(f"F-statistic = {F_stat_manual:.4f}")
print(f"p-value = {p_value_manual:.4f}")

# Perform F-test with SciPy ...................................................

# Calculate F statistic and p-value
F_stat_lib, p_value_lib = stats.f_oneway(sample1, sample2)

print(f"\nUsing SciPy F-test (F-oneway):")
print(f"F-statistic = {F_stat_lib:.4f}")
print(f"p-value = {p_value_lib:.4f}")