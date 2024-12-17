# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 2. Normality and multiple groups.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, norm

# Random seed
np.random.seed(123)

# Formulate hypotheses
# H0: Data follows a normal distribution
# H1: Data does not follow a normal distribution

# Generate random data sampling from gaussian
sample_size = 100
observed_data = np.random.normal(loc = 0, scale = 1, size = sample_size)
# observed_data = np.random.randint(0, 10, size = sample_size)

# Generate points normal distribution
x = np.linspace(observed_data.min(), observed_data.max(), 500)
y = norm.pdf(x, loc = np.mean(observed_data), scale = np.std(observed_data))

# Define bins and calculate observed frequencies
num_bins = 10
bin_edges = np.linspace(observed_data.min(), observed_data.max(), num_bins + 1)
hist, _ = np.histogram(observed_data, bins = bin_edges)

# Plot observations and gaussian fit
plt.figure(figsize = (8, 6))
plt.hist(observed_data, bins = num_bins, density = True, alpha = 0.6, color = "skyblue", label = "observed Data")
plt.plot(x, y, "r-", label = "Normal Distribution")
plt.title("Histogram of observations")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()

# Compute expected frequencies
cdf_values = norm.cdf(bin_edges, loc = np.mean(observed_data), scale = np.std(observed_data))
expected_frequencies = sample_size * np.diff(cdf_values)
# Normalize expected frequencies to match the sum of observed frequencies
expected_frequencies *= hist.sum() / expected_frequencies.sum()

# Compute Chi-square statistic manually
chi_square_manual = np.sum((hist - expected_frequencies) ** 2 / expected_frequencies)

# Compute Chi-square statistic using scipy
chi_square_scipy, p_value = chisquare(hist, f_exp = expected_frequencies)
print("Chi-square Statistic (scipy):", chi_square_scipy)
print("P-value:", p_value)

# Print results
print("Observed Frequencies:", hist)
print("Expected Frequencies:", np.round(expected_frequencies, 2))
print("Chi-square Statistic (manual):", chi_square_manual)
print("P-value:", p_value)

# Interpret result
alpha = 0.05
if p_value < alpha:
    print("\nReject H0: The data does not follow a normal distribution.")
else:
    print("Fail to reject H0: The data follows a normal distribution.")

# Plot the chi2 distribution
x = np.linspace(0, 30, 500)
df = num_bins - 1
y = norm.pdf(x, loc = chi_square_manual, scale = np.sqrt(2*df))
plt.figure(figsize = (8, 6))
plt.plot(x, y, label = "Chi-square Distribution")
plt.axvline(chi_square_manual, color = "r", linestyle = "--", label = f"Chi-square Observed: {chi_square_manual:.2f}")
plt.title("Chi-square Goodness-of-Fit Test")
plt.xlabel("Chi-square Value")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()
