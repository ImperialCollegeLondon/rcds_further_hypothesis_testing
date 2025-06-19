# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 2. Normality and multiple groups.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, f

# Random seed
np.random.seed(42)

# Formulate hypotheses
# H0: All data come from same distribution
# H1: At least one sample comes from different distribution

# Generate gaussian observations
group1 = np.random.normal(loc = 5, scale = 1, size = 10)
group2 = np.random.normal(loc = 5, scale = 1, size = 10)
group3 = np.random.normal(loc = 5, scale = 1, size = 10)
# group1 = np.random.normal(loc = 5, scale = 1, size = 100)
# group2 = np.random.normal(loc = 5, scale = 1, size = 100)
# group3 = np.random.normal(loc = 5, scale = 1, size = 100)
data = [group1, group2, group3]

# Plot observations
plt.figure(figsize = (8, 6))
plt.violinplot(data, showmeans = True, showextrema = True)
plt.title("Plot of observations")
plt.ylabel("Sample mean")
plt.show()

# Compute group means
mean1 = np.mean(group1)
mean2 = np.mean(group2)
mean3 = np.mean(group3)
overall_mean = np.mean(np.concatenate([group1, group2, group3]))

# Degrees of freedom
df_between = len(data) - 1
df_within = len(group1) + len(group2) + len(group3) - len(data)

# Correct SSB calculation
ssb = (len(group1) * (mean1 - overall_mean)**2 + 
       len(group2) * (mean2 - overall_mean)**2 + 
       len(group3) * (mean3 - overall_mean)**2)

# SSW stays the same
ssw = np.sum((group1 - mean1)**2) + np.sum((group2 - mean2)**2) + np.sum((group3 - mean3)**2)

# Mean squares
msb = ssb / df_between
msw = ssw / df_within

# F-statistic
F_stat_manual = msb / msw
p_value_manual = 1 - f.cdf(F_stat_manual, df_between, df_within)
print("\nANOVA (manual):")
print(f"F statistic (manual): {F_stat_manual:.4f}")
print(f"p value (manual): {p_value_manual:.4f}")

# SciPy ANOVA for verification
F_stat_scipy, p_value_scipy = f_oneway(group1, group2, group3)
print("\nANOVA (scipy stats):")
print(f"F statistic (scipy): {F_stat_scipy:.4f}")
print(f"p value (scipy): {p_value_scipy:.4f}")

# Conclusion based on the p-value
alpha = 0.05
if p_value_manual < alpha:
    print("\nReject the null hypothesis: There are significant differences between the groups.")
else:
    print("\nFail to reject the null hypothesis: No significant differences between the groups.")
