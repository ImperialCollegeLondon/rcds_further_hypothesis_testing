# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 3. Multiple hypothesis testig.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pdb


# List of p-values
p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
n = len(p_values)
np.set_printoptions(precision = 3)

# Bonferroni correction: Multiply each p-value by the number of tests
bonferroni_pvals = np.array(p_values) * n
bonferroni_pvals = np.clip(bonferroni_pvals, 0, 1)  # Ensures p-values do not exceed 1
print("\nBonferroni corrected:\n", bonferroni_pvals)

# Benjamini-Hochberg correction:

# Sort p-values and keep their original indices
sorted_indices = np.argsort(p_values)
sorted_pvals = np.array(p_values)[sorted_indices]

# Compute BH-adjusted p-values
bh_pvals = sorted_pvals * n / (np.arange(1, n + 1))

# Reorder BH-adjusted p-values back to the original order
bh_corrected_pvals = np.zeros(n)
bh_corrected_pvals[sorted_indices] = bh_pvals
print("\nBH corrected:\n", bh_corrected_pvals)

# Plotting the results
plt.figure(figsize = (8, 6))
plt.plot(p_values, label = "Original p-values", marker = 'o', linestyle = '-', color = 'blue')
plt.plot(bonferroni_pvals, label="Bonferroni corrected", marker = 'x', linestyle = '-', color = 'red')
plt.plot(bh_corrected_pvals, label="BH corrected", marker = 's', linestyle = '-', color = 'green')
bh_critical_threshold = np.linspace(0, 1, len(p_values))
plt.plot(bh_critical_threshold, linestyle = '--', color = 'black', label = "Critical BH threshold")
plt.xlabel("Rank (k)")
plt.ylabel("P-value")
plt.title("Bonferroni vs BH correction for p-values")
plt.legend()
plt.show()
