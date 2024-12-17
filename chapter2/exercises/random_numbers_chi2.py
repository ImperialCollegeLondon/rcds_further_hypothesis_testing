# RCDS Introduction to probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 2. Normality and multiple groups.

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Random seed
np.random.seed(123)

# Simulate random choices
def simulate_random_choices(n_people):

    # Uniform distribution
    random_choices = np.random.randint(0, 10, size = n_people * 3)
    return random_choices

# Simulate human-biased number choices
def simulate_human_choices(n_people):

    # Frequency vector for numbers 1 to 9
    frequencies = [1, 2, 1, 3, 1, 2, 4, 2, 1]
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

# Compare expected vs observed choices
print("\nCompare expected vs observed choices")

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
