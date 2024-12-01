# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - ICL London 2024 / 2025
# Chapter 3 - Bayesian statistics

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Random seed
np.random.seed(123)

# Function applying Bayes' Theorem
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Function computing probability of evidence
def total_probability(likelihoods, priors):
    return np.sum(np.array(likelihoods) * np.array(priors))

# Simulate multiple dice rolls ..................................................................
print("Simulate multiple dice rolls:\n")

# Simulation parameters
num_rolls = 50  # Number of dice rolls to simulate
biased_die_prob = 0.5  # Probability of rolling a 6 with the biased die
fair_die_prob = 1/6    # Probability of rolling a 6 with a fair die

# Initial priors
P_H1 = 0.5  # Prior: the die is biased
P_H2 = 0.5  # Prior: the die is fair

# True die: let's simulate a **biased die** 
true_prob = biased_die_prob  # True probability of rolling a 6

# Simulate the number of 6's in num_rolls trials using a binomial distribution
# This simulates whether each roll is a 6 or not based on biased probability
six_counts = np.random.binomial(1, true_prob, num_rolls)

# Lists to track the evolution of posterior probabilities
posterior_H1_values = [P_H1]
posterior_H2_values = [P_H2]

# Perform Bayesian updating for each roll
for i, outcome in enumerate(six_counts):

    # Store roll outcome
    if outcome == 1:
        roll_result = 6
    else:
        roll_result = "not 6"
    
    print(f"Roll {i+1}: {roll_result}")
    
    # Likelihood of the evidence (whether it's a 6 or not)
    if roll_result == 6:
        P_E_given_H1 = biased_die_prob  # Likelihood of rolling 6 on biased die
        P_E_given_H2 = fair_die_prob    # Likelihood of rolling 6 on fair die
    else:
        P_E_given_H1 = 1 - biased_die_prob  # Likelihood of NOT rolling a 6 on biased die
        P_E_given_H2 = 1 - fair_die_prob    # Likelihood of NOT rolling a 6 on fair die

    # Total probability of the evidence
    P_E = total_probability([P_E_given_H1, P_E_given_H2], [P_H1, P_H2])

    # Update posterior probabilities using Bayes' Theorem
    P_H1 = bayes_theorem(P_H1, P_E_given_H1, P_E)  # Posterior for H1 (biased die)
    P_H2 = 1 - P_H1  # Posterior for H2 (fair die)

    # Store the posterior probability for H1
    posterior_H1_values.append(P_H1)
    posterior_H2_values.append(P_H2)

    print(f"Updated Probability that the die is biased: {P_H1:.4f}\n")

# Plot the evolution of posterior probability for H1 (biased die)
plt.plot(posterior_H1_values, marker = "o", linestyle = "-", color = "b", label = "P(H1 | Evidence)")
plt.plot(posterior_H2_values, marker = "o", linestyle = "-", color = "r", label = "P(H2 | Evidence)")
plt.title("Evolution of posterior probability for biased die (H1) vs fair die (H2)")
plt.xlabel("Number of rolls")
plt.ylabel("Posterior probability")
plt.legend()
plt.grid(True)
plt.show()