## RCDS - Advanced probability & statistical inference

### Dr. Jesús Urtasun Elizari

### Imperial College London - 2024 / 2025

<img src="/readme_figures/grad-school-logo.png">

## Chapter 2. Normality & comparing multiple groups.

In this chapter we learn how to compare distributions and check for normaliry.
Finally we will see how to extend the t-tes and F-test to compare multiple groups, using ANOVA.

### ANOVA
 
Analysis of variances (ANOVA) for many groups.

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, f

# Random seed
np.random.seed(42)

# Generate gaussian observations
group1 = np.random.normal(loc = 5, scale = 1, size = 10)
group2 = np.random.normal(loc = 5, scale = 1, size = 10)
group3 = np.random.normal(loc = 5, scale = 1, size = 10)
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

# Copute sum of squares between (SSB) and within (SSW)
ssb = len(data) * ((mean1 - overall_mean)**2 + (mean2 - overall_mean)**2 + (mean3 - overall_mean)**2)
ssw = np.sum((group1 - mean1)**2) + np.sum((group2 - mean2)**2) + np.sum((group3 - mean3)**2)
df_between = len(data) - 1  # n - 1
df_within = len(group1) + len(group2) + len(group3) - len(data)  # N - n

# Mean Squares
msb = ssb / df_between
msw = ssw / df_within
df1 = len(data)
df2 = len(data)

# F-statistic
F_stat_manual = msb / msw
p_value_manual = 1 - f.cdf(F_stat_manual, df1, df2)
print("\nANOVA (manual):")
print(f"F statistic (manual): {F_stat_manual:.2f}")
print(f"p value (manual): {p_value_manual:.4f}")

# Compute ANOVA (scipy)
f_stat, p_value = f_oneway(group1, group2, group3)
print("\nANOVA (scipy stats):")
print(f"F statistic (scipy): {f_stat:.2f}")
print(f"p value (scipy): {p_value:.4f}")

# Conclusion based on the p-value
alpha = 0.05
if p_value_manual < alpha:
    print("\nReject the null hypothesis: There are significant differences between the groups.")
else:
    print("\nFail to reject the null hypothesis: No significant differences between the groups.")


```

### Chi2 for checking normality

Comparing a given set of observations with a hypothesized normal distribution.

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2, norm
from scipy.integrate import quad

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
df = num_bins - 1
p_value_manual, _ = quad(chi2.pdf, chi_square_manual, np.inf, args = (df, ))
print("\nChi-square Statistic (manual):", chi_square_manual)
print("P-value:", p_value_manual)

# Compute Chi-square statistic using scipy
chi_square_scipy, p_value = chisquare(hist, f_exp = expected_frequencies)
print("\nChi-square Statistic (scipy):", chi_square_scipy)
print("P-value:", p_value)

# Interpret result
alpha = 0.05
if p_value < alpha:
    print("\nReject H0: The data does not follow a normal distribution.")
else:
    print("\nFail to reject H0: The data follows a normal distribution.")

# Plot the chi2 distribution
x = np.linspace(0, 30, 500)
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

```

### Chi2 for comparing distributions

Comparing a given set of observations with some hypothesized distribution.

```python

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Simulate random choices
def simulate_random_choices(n_people):

    # Uniform distribution
    random_choices = np.random.randint(0, 10, size = n_people * 3)
    return random_choices

# Simulate human-biased number choices
def simulate_human_choices(n_people):

    # Frequency vector for numbers 1 to 9
    frequencies = [1, 2, 1, 2, 1, 2, 1, 2, 1]
    total_frequency = sum(frequencies)
    
    # Normalize frequencies / compute probabilities
    human_bias = np.array(frequencies) / total_frequency
    
    # Add 0 with small frequency to complete bias for numbers 0-9
    human_bias = np.insert(human_bias, 0, (1 / total_frequency))
    
    # Check sum of probabilities equals 1
    human_bias /= human_bias.sum()
    
    # Simulate choices using the computed bias (each person picks 3 numbers)
    choices = np.random.choice(range(10), size = n_people * 3, p = human_bias)
    return choices

# Random seed
np.random.seed(123)

# Compare expected vs observed choices
print("\nCompare expected vs observed choices")

# Formulate hypotheses
# H0: Data follows a uniform distribution
# H1: Significant difference from uniform distribution

# Prepare expected and observed choices
n_people = 10 # Number of participants
human_choices = simulate_human_choices(n_people)
random_choices = simulate_random_choices(n_people)

# Count frequencies
random_freq =[np.sum(random_choices == i) for i in range(10)]
human_freq = [np.sum(human_choices == i) for i in range(10)]

# Perform chi-square test
expected_freq = [n_people * 3 / 10] * 10  # Expected frequency for uniform distribution
chi2_stat, p_value = chisquare(human_freq, f_exp = expected_freq)

# Plot the results
x_labels = [str(i) for i in range(10)]
x = np.arange(len(x_labels))
plt.figure(figsize = (10, 6))
# Plot human (observed) choices
plt.bar(x - 0.2, human_freq, width = 0.4, label = "Human Choices", color = "skyblue")
# Plot random (expected) choices
plt.bar(x + 0.2, random_freq, width = 0.4, label = "Random Choices", color = "orange")
plt.axhline(y = n_people * 3 / 10, color = "gray", linestyle = "--", label = "Expected Uniform Frequency")
# Add lables and title
plt.title("Distribution of Numbers: Human Choices vs Random Choices")
plt.xlabel("Numbers")
plt.ylabel("Frequency")
plt.xticks(x, x_labels)
plt.legend()
plt.show()

# Print results
print("Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\nThe observed distribution significantly deviates from a uniform distribution.")
else:
    print("\nNo significant deviation from a uniform distribution.")

```