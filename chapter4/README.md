## RCDS - Advanced probability & statistical inference

### Dr. Jesús Urtasun Elizari

### Imperial College London - 2024 / 2025

<img src="/readme_figures/grad-school-logo.png">

## Chapter 4. Introduction to bayesian statstics.

In this chapter we will introduce the Bayes theorem, and bayesian statistics.
We will describe the *prior* and *posterior* distributions, and learn how to use the Bayes rule to update the probability given certain evidence.
Import the libraries neeed for these examples.

```python

# Import libraries
import numpy as np
from math import comb, exp, factorial, erf, sqrt
from scipy import stats
import matplotlib.pyplot as plt

```

We will use the following functions:

```python

# Function applying Bayes' Theorem
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Function computing probability of evidence
def total_probability(likelihoods, priors):
    return np.sum(np.array(likelihoods) * np.array(priors))

```

### Bayesian probability

Example of rolling dice:

```python

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


```

Example of medical experiment:

```python

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

```

### Bayesian statistics

Simulate multiple dice rolls:

```python

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

```