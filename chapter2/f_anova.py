# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, f

# Random seed
np.random.seed(1234)

# Formulate hypothesis
# H0: All data comes from same distribution
# H1: At least one sample comes from different distribution

# Generate gaussian observations
group1 = np.random.normal(loc = 5, scale = 1, size = 10)
group2 = np.random.normal(loc = 8, scale = 1, size = 10)
group3 = np.random.normal(loc = 5, scale = 1, size = 10)
data = [group1, group2, group3]
print("Observations:\ngroup1: ", data[0], "\ngroup2: ", data[1], "\ngroup3: ", data[2])

# Plot observations
plt.figure(figsize = (8, 6))
plt.violinplot(data, showmeans = True, showextrema = True)
plt.title("Plot of observations")
plt.ylabel("Sample mean")
plt.show()

# Compute group means
mean1, mean2, mean3 = np.mean(data[0]), np.mean(data[1]), np.mean(data[2])
overall_mean = np.mean(np.concatenate(data))
print("\nmean1: ", mean1, "\nmean2: ", mean2, "\nmean3: ", mean3)

# Degrees of freedom
df_between = len(data) - 1
df_within = len(group1) + len(group2) + len(group3) - len(data)

# Variation between groups (SSB)
ssb = (len(group1) * ((mean1 - overall_mean))**2 + 
    len(group2) * ((mean2 - overall_mean))**2 +
    len(group3) * ((mean3 - overall_mean))**2)

# Variation within groups (SSW)
ssw = np.sum((group1 - mean1)**2) + np.sum((group2 - mean2)**2) + np.sum((group3 - mean3)**2)

# Normalize with degrees of freedom
ssb = ssb / df_between
ssw = ssw / df_within

# F-statistic manual calculation
F_stat_manual = ssb / ssw
p_value_manual = 1 - f.cdf(F_stat_manual, df_between, df_within)
print("\nF ANOVA (manual): ", F_stat_manual)
print("p-value (manual): ", p_value_manual)

# F-statistic with scipy library
F_stat_scipy, p_value_scipy = f_oneway(group1, group2, group3)
print("\nF ANOVA (scipy): ", F_stat_scipy)
print("p-value (scipy): ", p_value_scipy)