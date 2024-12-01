## RCDS - Advanced probability & statistical inference

### Dr. Jesús Urtasun Elizari

### Imperial College London - 2024 / 2025

<img src="/readme_figures/grad-school-logo.png">

## Chapter 2. Normality & comparing multiple groups.

In this chapter we learn how to compare distributions and check for normaliry.
Finally we will see how to extend the t-tes and F-test to compare multiple groups, using ANOVA.
Import the libraries neeed for these examples.

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

```

### Chi2 for comparing distributions

Chi2 for comparing distributions:

```python

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

```

### Chi2 for checking normality

Chi2 for checking normality:

```python

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

```

### Comparing multiple groups with ANOVA

Comparing multiple groups with ANOVA:

```python

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

# F-statistic
f_statistic = msb / msw

# Print results
print("\nANOVA (manual):")
print(f"Sum of Squares Between (SSB): {ssb:.2f}")
print(f"Sum of Squares Within (SSW): {ssw:.2f}")
print(f"Mean Square Between (MSB): {msb:.2f}")
print(f"Mean Square Within (MSW): {msw:.2f}")
print(f"F-statistic: {f_statistic:.2f}")

# Compute ANOVA (scipy)
f_stat, p_value = stats.f_oneway(group1, group2, group3)

# Print results
print("\nANOVA (scipy stats):")
print(f"F-statistic (scipy): {f_stat:.2f}")
print(f"P-value (scipy): {p_value:.4f}")

# Conclusion based on the p-value
alpha = 0.05
if p_value < alpha:
    print("\nReject the null hypothesis: There are significant differences between the groups.")
else:
    print("\nFail to reject the null hypothesis: No significant differences between the groups.")

```