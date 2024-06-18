# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - Imperial College London
# Chapter 1 - Two sample t-test

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Formulate null and alternative hypothesis
# H0: mean1, mean2 come from same distribution
# H1: mean1, mean2 com from different distibutions
alpha = 0.05 # significance level

# Generate random data
np.random.seed(123)
num_samples = 100
data = {
    "Column1": np.random.normal(loc = 60, scale = 10, size = num_samples),
    "Column2": np.random.normal(loc = 60, scale = 15, size = num_samples)
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Plot to visualize data
plt.figure(figsize = (8, 5))
plt.hist(df["Column1"], bins = 20, color = "skyblue", alpha = 0.5, label = "Sample 1")
plt.hist(df["Column2"], bins = 20, color = "lightgreen", alpha = 0.5, label = "Sample 2")
plt.xlabel("value")
plt.ylabel("frequency")
plt.title("Histograms of sample data")
plt.grid(axis = "y", alpha = 0.75)
plt.legend()
plt.show()

# Compute means and standard deviations
n1 = len(df["Column1"])
n2 = len(df["Column2"])
mean1 = df["Column1"].mean()
mean2 = df["Column2"].mean()
std1 = df["Column1"].std()
std2 = df["Column2"].std()
print("\nMean value of Column1: ", mean1, "std1: ", std1)
print("Mean value of Column2: ", mean2, "std2", std2)

# Two sample t-test: manual
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
t_stat_manual = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))
dof = n1 + n2 - 2
p_value_manual = 2 * (1 - stats.t.cdf(abs(t_stat_manual), dof))
print("\nManual t-test:\nt-statistic:", t_stat_manual, "p-value: ", p_value_manual)

# Two-sample t-test:  scipy
t_stat_scipy, p_value_scipy = stats.ttest_ind(df["Column1"], df["Column2"], equal_var = True)
print("Scipy t-test:\nt-statistic:", t_stat_scipy, "p-value:", p_value_scipy)