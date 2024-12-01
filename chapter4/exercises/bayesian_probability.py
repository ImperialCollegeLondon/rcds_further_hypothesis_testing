# RCDS Further hypothesis testing
# Jesus Urtasun Elizari - ICL London 2024 / 2025
# Chapter 3 - Bayesian statistics

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Function applying Bayes' Theorem
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Function computing probability of evidence
def total_probability(likelihoods, priors):
    return np.sum(np.array(likelihoods) * np.array(priors))


# Example 1: rolling dice..........................................................................
print("Example 1: rolling dice:\n")

# Initial priors
P_H1 = 0.5  # Prior: biased die
P_H2 = 0.5  # Prior: fair die
print(f"Prior probabilities:\nDie is biased, P(H1) = {P_H1}\nDie is fair, P(H2) = {P_H2}\n")

# First roll: rolled a 6
P_E_given_H1 = 1/2  # Likelihood of rolling 6 on biased die
P_E_given_H2 = 1/6  # Likelihood of rolling 6 on fair die

# Total probability of evidence
P_E = total_probability([P_E_given_H1, P_E_given_H2], [P_H1, P_H2])

# Posterior probability after first roll
P_H1_given_E1 = bayes_theorem(P_H1, P_E_given_H1, P_E)
print(f"After 1st roll of 6: Probability that the die is biased: {P_H1_given_E1:.2f}")

# Update priors for second roll
P_H1 = P_H1_given_E1
P_H2 = 1 - P_H1

# Second roll: rolled another 6
P_E2_given_H1 = 0.5  # Likelihood of rolling another 6 on biased die
P_E2_given_H2 = 1/6  # Likelihood of rolling another 6 on fair die

# Total probability of second evidence
P_E2 = total_probability([P_E2_given_H1, P_E2_given_H2], [P_H1, P_H2])

# Posterior probability after second roll
P_H1_given_E2_E1 = bayes_theorem(P_H1, P_E2_given_H1, P_E2)
print(f"After 2nd roll of 6: Probability that the die is biased: {P_H1_given_E2_E1:.2f}\n")


# Example 2: medical test .........................................................................
print("Example 2: medical test:\n")

# Initial priors
P_H = 0.01  # Prior: having the disease
P_not_H = 0.99  # Prior: not having the disease
print(f"Prior probabilities:\nHave disease, P(H1) = {P_H}\nNot have disease, P(H2) = {P_not_H}\n")

# First test: tested positive
P_E_given_H = 0.99  # Likelihood of testing positive given disease
P_E_given_not_H = 0.01  # Likelihood of testing positive without disease

# Total probability of evidence (first test result)
P_E = total_probability([P_E_given_H, P_E_given_not_H], [P_H, P_not_H])

# Posterior probability after first test
P_H_given_E1 = bayes_theorem(P_H, P_E_given_H, P_E)
print(f"After 1st positive test: Probability of having the disease: {P_H_given_E1:.2f}")

# Update priors for second test
P_H = P_H_given_E1
P_not_H = 1 - P_H

# Second test: tested positive again
P_E2_given_H = 0.99  # Likelihood of testing positive again given disease
P_E2_given_not_H = 0.01  # Likelihood of testing positive again without disease

# Total probability of second evidence (second test result)
P_E2 = total_probability([P_E2_given_H, P_E2_given_not_H], [P_H, P_not_H])

# Posterior probability after second test
P_H_given_E2_E1 = bayes_theorem(P_H, P_E2_given_H, P_E2)
print(f"After 2nd positive test: Probability of having the disease: {P_H_given_E2_E1:.2f}\n")
