# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 1. Parameter estimation and hypotesis testing.

# Import libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pdb

# Read input data
data = pd.read_csv('data/data_stars.csv', index_col = False)
print(data.head())

pdb.set_trace()

# Separate data into two groups based on spectral class (group 1, O and B; group 2, G and K )
group1 = data[data['Class'].isin(['O', 'B'])]['Luminosity'].values
group2 = data[data['Class'].isin(['G', 'K'])]['Luminosity'].values

# Calculate means of each group
mean1 = np.var(group1)
mean2 = np.var(group2)

# Plot histograms
plt.figure(figsize = (10, 6))
plt.hist(group1, bins = 15, alpha = 0.6, color = 'blue', label = f'Group 1 (Mean: {mean1:.2f})')
plt.hist(group2, bins = 15, alpha = 0.6, color = 'red', label = f'Group 2 (Mean: {mean2:.2f})')
# Highlight the variances
plt.axvline(mean1, color = 'blue', linestyle = 'dashed', linewidth = 1.5, label=f'Group 1 Mean = {mean1:.2f}')
plt.axvline(mean2, color = 'red', linestyle = 'dashed', linewidth = 1.5, label=f'Group 2 Mean = {mean2:.2f}')
# Customize the plot
plt.title('Luminosity Distributions for Two Groups')
plt.xlabel('Luminosity')
plt.ylabel('Frequency')
plt.legend()
# Show plot
plt.show()

# Perform Fisher test
f_statistic, p_value = stats.levene(class_a_luminosity, class_b_luminosity)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Interpret result
if p_value < 0.05:
    print("The variances are significantly different (reject the null hypothesis).")
else:
    print("The variances are not significantly different (fail to reject the null hypothesis).")