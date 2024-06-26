# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - Imperial College London
# Chapter 1 - Fisher test

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Formulate null and alternative hypothesis
# H0: var1, var2 come from same distribution
# H1: var1, var2 com from different distibutions
alpha = 0.05 # significance level

# Generate random data
np.random.seed(123)
num_samples = 100
data = {
    "Column1": np.random.normal(loc = 50, scale = 10, size = num_samples),
    "Column2": np.random.normal(loc = 60, scale = 10, size = num_samples)
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Plot to visualize data
plt.figure(figsize = (8, 5))
plt.hist(df["Column1"], bins = 20, color = "skyblue", alpha = 0.5, label = "Sample 1")
plt.hist(df["Column2"], bins = 20, color = "lightgreen", alpha = 0.5, label = "Sample 2")
plt.xlabel("value")
plt.ylabel("frequency")
plt.title("Histograms of Sample Data")
plt.grid(axis = "y", alpha = 0.75)
plt.legend()
plt.show()

# Compute the variances of the two groups
var1 = np.var(df["Column1"], ddof = 1)
var2 = np.var(df["Column2"], ddof = 1)
print("\nVariance of Column1: ", var1)
print("Variance of Column2: ", var2)

# Perform the F-test to compare variances
f_statistic = var1 / var2
p_value = stats.f.cdf(f_statistic, len(df["Column1"])-1, len(df["Column2"])-1)
print("\nF-statistic:", f_statistic)
print("p-value:", p_value)