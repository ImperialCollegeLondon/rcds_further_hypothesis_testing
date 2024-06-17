# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - Imperial College London
# Chapter 1 - Comparing means

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data
np.random.seed(123)
num_samples = 100
data = {
    "Column1": np.random.normal(loc = 60, scale = 10, size = num_samples),
    "Column2": np.random.normal(loc = 60, scale = 15, size = num_samples)
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Plot the histograms with transparency
plt.figure(figsize = (8, 5))
plt.hist(df["Column1"], bins = 20, color = "skyblue", alpha = 0.5, label = "Sample 1")
plt.hist(df["Column2"], bins = 20, color = "lightgreen", alpha = 0.5, label = "Sample 2")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histograms of Sample Data")
plt.legend()
plt.show()

# Define the null hypothesis and significance level
# H0: mean1, mean2 come from same distribution
alpha = 0.05 # h0_mean = 45; # For last exercise

# Compute mean values
mean1 = df["Column1"].mean()
mean2 = df["Column2"].mean()
print("\nMean value of Column1: ", mean1)
print("Mean value of Column2: ", mean2)

# Perform two-sample t-test assuming equal variances
t_statistic, p_value = stats.ttest_ind(df["Column1"], df["Column2"], equal_var = True)
print("\nt-statistic:", t_statistic)
print("p-value:", p_value)

# # Perform a one-tailed t-test
# t_statistic_one_tailed, p_value_one_tailed = stats.ttest_1samp(df, h0_mean, alternative = "greater")
# print("\nOne-tailed t-test:")
# print("t-statistic:", t_statistic_one_tailed[0])
# print("p-value:", p_value_one_tailed[1])

# # Perform a two-tailed t-test
# t_statistic_two_tailed, p_value_two_tailed = stats.ttest_1samp(df, h0_mean)
# print("\nTwo-tailed t-test:")
# print("t-statistic:", t_statistic_two_tailed[0])
# print("p-value:", p_value_two_tailed[1])