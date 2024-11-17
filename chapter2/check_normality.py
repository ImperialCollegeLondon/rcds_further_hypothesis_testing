# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 2. Normality and multiple groups.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2_contingency

# Random seed
np.random.seed(42)

# Parameters for distribution
mean = 0
std = 1
size = 1000

# Sample from gaussian distiribution
data = np.random.normal(mean, std, size)

# Plot histogram and gaussian function
bins = np.linspace(-4, 4, 30)  # Define bin edges for histogram
bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Compute bin centers for plotting the PDF

# Histogram of the data
hist, _ = np.histogram(data, bins = bins, density = True)

# Theoretical Gaussian PDF
pdf = norm.pdf(bin_centers, mean, std)

# Plot the histogram and theoretical Gaussian
plt.figure(figsize = (10, 6))
plt.hist(data, bins = bins, density = True, alpha = 0.6, label = "Sampled Data")
plt.plot(bin_centers, pdf, label = "Gaussian PDF (mean = 0, std = 1)", color = "red", linewidth = 2)
plt.title("Gaussian Data vs. Gaussian Function")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# Scale histogram to match the sample size for chi-squared test
observed_counts, _ = np.histogram(data, bins = bins)
expected_counts = size * np.diff(bins) * pdf  # Scale pdf to expected counts
# Add a small constant to avoid zeros
contingency_table = np.array([observed_counts, expected_counts + 1e-10])
print("\nContingency Table (Observed vs. Expected):\n", contingency_table)
print("Dimensions: ", contingency_table.shape)

# Perform chi-squared test
chi2, p_value = chi2_contingency([observed_counts, expected_counts + 1e-10])[:2]

# Print results
print(f"\nChi-squared test statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("The data significantly deviates from a Gaussian distribution (reject H0).")
else:
    print("The data does not significantly deviate from a Gaussian distribution (fail to reject H0).")