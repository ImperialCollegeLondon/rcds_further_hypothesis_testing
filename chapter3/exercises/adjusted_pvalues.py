# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 3. Multiple hypothesis testing.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# List of p-values
p_values = [0.01, 0.02, 0.03, 0.04, 0.10]
n = len(p_values)
np.set_printoptions(precision=3)
print("\nOriginal p-values:", p_values)

# Bonferroni correction
bonferroni_pvals = np.array(p_values) * n
bonferroni_pvals = np.clip(bonferroni_pvals, 0, 1)  # Ensures p-values do not exceed 1
print("\nBonferroni corrected:", bonferroni_pvals)

# Benjamini-Hochberg correction (Fixed)
sorted_indices = np.argsort(p_values)
sorted_pvals = np.array(p_values)[sorted_indices]

# Compute BH-adjusted p-values
bh_pvals = sorted_pvals * n / (np.arange(1, n + 1))
# Step-down adjustment to ensure monotonicity
bh_pvals = bh_pvals[::-1]  # Reverse the order
bh_pvals = np.minimum.accumulate(bh_pvals)  # Apply cumulative minimum
bh_pvals = bh_pvals[::-1]  # Reverse back to original order  # Step-down adjustment

# Reorder BH-adjusted p-values back to original order
bh_corrected_pvals = np.zeros(n)
bh_corrected_pvals[sorted_indices] = bh_pvals
print("\nBH corrected:", bh_corrected_pvals)

# Plotting results
plt.figure(figsize=(8, 6))
plt.plot(range(1, n + 1), p_values, label="Original p-values", marker='o', linestyle='-', color='blue')
plt.plot(range(1, n + 1), bonferroni_pvals, label="Bonferroni corrected", marker='x', linestyle='-', color='red')
plt.plot(range(1, n + 1), bh_corrected_pvals, label="BH corrected", marker='s', linestyle='-', color='green')
plt.xlabel("Test Rank")
plt.ylabel("P-value")
plt.title("Bonferroni vs BH correction for p-values")
plt.legend()
plt.show()
