# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 2. Normality and multiple groups.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Random seed
np.random.seed(42)

# Parameters for distribution 1
mean1 = 0
std1 = 1
size1 = 1000

# Parameters for distribution 2
mean2 = 0.5
std2 = 1.2
size2 = 1000

# Sample from gaussian distribution
data1 = np.random.normal(mean1, std1, size1)
data2 = np.random.normal(mean2, std2, size2)

# Plot the distributions
plt.figure(figsize = (10, 6))
bins = np.linspace(-5, 5, 30)

# Use common bins for histogram
plt.hist(data1, bins, alpha = 0.5, label = "Distribution 1 (mean = 0, std = 1)", density = True)
plt.hist(data2, bins, alpha = 0.5, label = "Distribution 2 (mean = 0.5, std = 1.2)", density = True)
plt.title("Gaussian distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# Convert data into histogram counts for the chi-squared test
hist1, bin_edges1 = np.histogram(data1, bins = bins)
hist2, bin_edges2 = np.histogram(data2, bins = bins)

# Combine histograms into a contingency table
contingency_table = np.array([hist1, hist2])
# Add a small constant to avoid zero bins
contingency_table = np.array([hist1 + 1, hist2 + 1])
print("\ndata1: ", hist1, "\ndata2: ", hist2)
print("\nContingency Table (distribution 1 vs distribution 2):\n", contingency_table)
print("Dimensions: ", contingency_table.shape)

# Perform the Chi-squared test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Print results
print(f"\nChi-squared test statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("The distributions are significantly different (reject H0).")
else:
    print("No significant difference between the distributions (fail to reject H0).")