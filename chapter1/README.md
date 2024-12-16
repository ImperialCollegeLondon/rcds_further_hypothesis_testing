## RCDS - Advanced probability & statistical inference

### Dr. Jesús Urtasun Elizari

### Imperial College London - 2024 / 2025

<img src="/readme_figures/grad-school-logo.png">

## Chapter 1. Parameter estimation & hypothesis testing.

In this chapter we will learn how to compare means and variances.
We will introduce for that the two-sample t-test, and the Fisher F-test.
Import the libraries neeed for these examples.

```python

# Import libraries
import numpy as np
from math import comb, exp, factorial, erf, sqrt
from scipy import stats
import matplotlib.pyplot as plt

```

### One sample t-test

Compare sample mean to hypothesized value.

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, t

# Random seed
np.random.seed(42)

# One sample t-test
print("\nOne sample t-test:\nCompare sample mean to hypothesized value")

# Formulate hypotheses
# H0 (null hypothesis): The coin is fair (p = 0.5)
# H1 (alternative hypothesis): The coin is biased (p != 0.5)

# Simulate 100 coin tosses (1 = heads, 0 = tails)
observations = np.random.choice([0, 1], size = 100, p = [0.5, 0.5])
# observations = np.random.choice([0, 1], size = 100, p = [0.3, 0.7])

# Plot histogram of the observations
plt.hist(observations, bins = 2, edgecolor = "black", alpha = 0.75)
plt.xticks([0.25, 0.75], ["Tails (0)", "Heads (1)"])
plt.title("Histogram of 100 coin tosses")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.show()

# Compute observed mean and standard deviation
n = len(observations)
mean_observed = np.mean(observations)
std_dev_observed = np.std(observations, ddof = 1)  # Sample standard deviation

# Compute the t statistic and p-value Expected mean under H0 = 0.5
expected_mean = 0.5
t_stat_manual = (mean_observed - expected_mean) / (std_dev_observed / np.sqrt(n))
p_value_manual = 2 * (1 - t.cdf(abs(t_stat_manual), df = n-1))
print(f"\nT statistic (manual): {t_stat_manual:.4f}")
print(f"P-value (manual): {p_value_manual:.4f}")

# Compute the t statistic using a library
t_stat_scipy, p_value_scipy = ttest_1samp(observations, popmean = 0.5)
print(f"\nUsing SciPy library:")
print(f"\nT statistic (library): {t_stat_scipy:.4f}")
print(f"P-value (library): {p_value_scipy:.4f}")

# Parameters for plot t-distribution
x = np.linspace(-4, 4, 1000) # Range for t-distribution
t_dist = t.pdf(x, df = n - 1) # Probability density function for t-distribution

# Plot t-distribution
plt.plot(x, t_dist, label = f"t-distribution (df = {n - 1})")
plt.axvline(t_stat_manual, color = "red", linestyle = "--", label = f"T statistic = {t_stat_manual:.2f}")
plt.axvline(-t_stat_manual, color = "red", linestyle = "--")
plt.fill_between(x, t_dist, where = (x >= abs(t_stat_manual)), color = "red", alpha = 0.3, label = "Rejection region (one side)")
plt.fill_between(x, t_dist, where = (x <= -abs(t_stat_manual)), color = "red", alpha = 0.3)
plt.title("t-distribution and observed t statistic")
plt.xlabel("t value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Interpret result (significance level 0.05)
alpha = 0.05
if p_value_manual < alpha:
    print("Reject H0: The coin is likely biased.")
else:
    print("Accept H0: No evidence of biased coin.")

```

### Two sample t-test:

Compare sample means of two independent groups.

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, t

# Random seed
np.random.seed(0)

# Two sample t-test
print("\nTwo sample t-test:\nCompare sample means of two independent groups")

# Formulate hypotheses
# H0: x1, x2 come from sampe distribution (mean1 = mean2 = mu)
# H1: x1, x2, come from different distriubtions (mean1 != mean2 != mu)

# Simulate two sets of gaussian observations
sample1 = np.random.normal(loc = 50, scale = 5, size = 30)
sample2 = np.random.normal(loc = 55, scale = 5, size = 30)

# Plot observations as histograms
plt.hist(sample1, bins = 15, alpha = 0.5, color = "blue", label = "Sample 1")
plt.hist(sample2, bins = 15, alpha = 0.5, color = "red", label = "Sample 2")
plt.title("Histogram of gaussian observations")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Compute the sample means and standard deviations
mean1, mean2 = np.mean(sample1), np.mean(sample2)
std1, std2 = np.std(sample1, ddof = 1), np.std(sample2, ddof = 1)
n1, n2 = len(sample1), len(sample2)
# Compute standard error of the difference
se = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

# Compute the t-statistic
t_stat_manual = (mean1 - mean2) / se
p_value_manual = 2 * (1 - t.cdf(abs(t_stat_manual), df = n1 + n2 - 2))
print(f"Manual calculation:")
print(f"t-statistic = {t_stat_manual:.4f}")
print(f"p-value = {p_value_manual:.4f}")

# Compute two-tailed statistic and p-value
t_stat_scipy, p_value_scipy = ttest_ind(sample1, sample2)
print(f"\nUsing SciPy library:")
print(f"t-statistic = {t_stat_scipy:.4f}")
print(f"p-value = {p_value_scipy:.4f}")

# Parameters for plot t-distribution
x = np.linspace(-4, 4, 1000) # Range for t-distribution
t_dist = t.pdf(x, df = n2 + n1 - 2) # Probability density function for t-distribution

# Plot t-distribution
plt.plot(x, t_dist, label = f"t-distribution (df = {n2 + n1 -1})")
plt.axvline(t_stat_manual, color = "red", linestyle = "--", label = f"T statistic = {t_stat_manual:.2f}")
plt.axvline(-t_stat_manual, color = "red", linestyle = "--")
plt.fill_between(x, t_dist, where = (x >= abs(t_stat_manual)), color = "red", alpha = 0.3, label = "Rejection Region (one side)")
plt.fill_between(x, t_dist, where = (x <= -abs(t_stat_manual)), color = "red", alpha = 0.3)
plt.title("t-distribution and observed t statistic")
plt.xlabel("t value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Interpret result (significance level 0.05)
alpha = 0.05
if p_value_manual < alpha:
    print("Reject H0: Samples come from different distributions.")
else:
    print("Accept H0: Samples come from same distribution.")

```

### Fisher test

Compare variances of two different groups

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, levene, bartlett, f

# Random seed
np.random.seed(42)

# Fisher test
print("\nFisher test:\nCompare sample variances of two independent groups")

# Simulate two sets of gaussian observations
sample1 = np.random.normal(loc = 50, scale = 5, size = 30)
sample2 = np.random.normal(loc = 55, scale = 7, size = 30)

# Plot observations as histograms
plt.hist(sample1, bins = 15, alpha = 0.5, color = "blue", label = "Sample 1")
plt.hist(sample2, bins = 15, alpha = 0.5, color = "red", label = "Sample 2")
plt.title("Histogram of gaussian Observations")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Calculate variances of the two samples
var1, var2 = np.var(sample1, ddof = 1), np.var(sample2, ddof = 1)
# Degrees of freedom for each sample
df1 = len(sample1) - 1
df2 = len(sample2) - 1

# Calculate F-statistic and p-value manually
F_stat_manual = var1 / var2 if var1 > var2 else var2 / var1
p_value_manual = 2 * (1 - f.cdf(F_stat_manual, df1, df2))
print(f"Manual calculation:")
print(f"F-statistic = {F_stat_manual:.4f}")
print(f"p-value = {p_value_manual:.4f}")

# # Calculate F statistic and p-value using library
# F_stat_scipy = max(var1, var2) / min(var1, var2)
# p_value_scipy = 2 * (1 - f.cdf(F_stat_scipy, df1, df2))
# print(f"\nUsing SciPy library:")
# print(f"F-statistic = {F_stat_scipy:.4f}")
# print(f"p-value = {p_value_scipy:.4f}")

# Plot the F-distribution
x = np.linspace(0, 5, 1000)  # Range of F-values for plotting
f_dist = f.pdf(x, df1, df2)  # F-distribution density function

# Plot F-distribution
plt.plot(x, f_dist, label = f"F-distribution (df1 = {df1}, df2 = {df2})")
plt.axvline(F_stat_manual, color = "red", linestyle = "--", label = f"Observed F = {F_stat_manual:.2f}")
plt.fill_between(x, f_dist, where = (x >= F_stat_manual), color = "red", alpha = 0.3, label = "Rejection region")
plt.title("F-distribution and observed F-statistic")
plt.xlabel("F value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Interpret result (significance level 0.05)
alpha = 0.05
if p_value_manual < alpha:
    print("Reject H0: Samples come from different distribution.")
else:
    print("Accept H0: Samples come from same distribution.")

```