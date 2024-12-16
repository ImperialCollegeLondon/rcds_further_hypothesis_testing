# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 2. Normality and multiple groups.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, f

# Random seed
np.random.seed(42)

# Generate gaussian observations
group1 = np.random.normal(loc = 2, scale = 1, size = 10)
group2 = np.random.normal(loc = 5, scale = 1, size = 10)
group3 = np.random.normal(loc = 8, scale = 1, size = 10)
# # Generate identical groups
# group1 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
# group2 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
# group3 = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

# Combine groups into a single array
data = [group1, group2, group3]

# Step 2: Plot the observations
plt.figure(figsize = (8, 6))
plt.boxplot(data, labels = ["Group 1", "Group 2", "Group 3"])
plt.title("Boxplot of observations")
plt.ylabel("Value")
plt.show()

# Compute ANOVA (manually)

# Group means
mean1 = np.mean(group1)
mean2 = np.mean(group2)
mean3 = np.mean(group3)
overall_mean = np.mean(np.concatenate([group1, group2, group3]))

# Sum of Squares Between (SSB)
ssb = len(group1) * ((mean1 - overall_mean)**2 + (mean2 - overall_mean)**2 + (mean3 - overall_mean)**2)

# Sum of Squares Within (SSW)
ssw = np.sum((group1 - mean1)**2) + np.sum((group2 - mean2)**2) + np.sum((group3 - mean3)**2)

# Degrees of freedom
df_between = len(data) - 1  # k - 1
df_within = len(group1) + len(group2) + len(group3) - len(data)  # N - k

# Mean Squares
msb = ssb / df_between
msw = ssw / df_within
df1 = len(data)
df2 = len(data)

# F-statistic
F_stat_manual = msb / msw
p_value_manual = 2 * (1 - f.cdf(F_stat_manual, df1, df2))

# Print results
print("\nANOVA (manual):")
print(f"Sum of Squares Between (SSB): {ssb:.2f}")
print(f"Sum of Squares Within (SSW): {ssw:.2f}")
print(f"Mean Square Between (MSB): {msb:.2f}")
print(f"Mean Square Within (MSW): {msw:.2f}")
print(f"F statistic (manual): {F_stat_manual:.2f}")
print(f"p value (manual): {p_value_manual:.4f}")

# Compute ANOVA (scipy)
f_stat, p_value = f_oneway(group1, group2, group3)

# Print results
print("\nANOVA (scipy stats):")
print(f"F statistic (scipy): {f_stat:.2f}")
print(f"P value (scipy): {p_value:.4f}")

# Conclusion based on the p-value
alpha = 0.05
if p_value < alpha:
    print("\nReject the null hypothesis: There are significant differences between the groups.")
else:
    print("\nFail to reject the null hypothesis: No significant differences between the groups.")
