# RCDS Advanced probability and statistical inference.
# Jesús Urtasun Elizari. ICL 2024 / 2025.
# Chapter 4. Bayesian statistics.

# Example 1
print("Example 2")

# Define hypotheses
# H1: "The die is biased"
# H2: "The die is fair"
# E: "I rolled a 6"

# Priors
P_H1 = 0.5  # Prior probability of rolling a biased die
P_H2 = 0.5  # Prior probability of rolling a fair die

# Likelihoods
P_E_given_H1 = 1/2 # P(E | H1): Probability of rolling a 6 given the die is biased
P_E_given_H2 = 1/6 # P(E | H2): Probability of rolling a 6 given the die is fair

# Compute marginal probability P(E):
# P(E) = P(E | H1) * P(H1) + P(E | H2) * P(H2)
P_E = P_E_given_H1 * P_H1 + P_E_given_H2 * P_H2

# Compute posterior probabilities using Bayes' rule
# P(H1 | E) = P(E | H1) * P(H1) / P(E)
P_H1_given_E = (P_E_given_H1 * P_H1) / P_E

# P(H2 | E) = P(E | H2) * P(H2) / P(E)
P_H2_given_E = (P_E_given_H2 * P_H2) / P_E

# Print the results
print(f"\nP(E) (Marginal probability of observing evidence): {P_E:.2f}")
print(f"P(H1|E) (Posterior probability of biased die given 6): {P_H1_given_E:.2f}")

# Example 2
print("\nExample 2:")

# Define hypotheses
# H1: "I have the disease"
# H2: "I don't have disease"
# E: "I tested positive"

# Given values
P_disease = 0.01  # Prevalence of the disease
P_positive_given_disease = 0.99  # Sensitivity (True Positive Rate)
P_positive_given_no_disease = 0.01  # False Positive Rate

# Probability of not having the disease
P_no_disease = 1 - P_disease

# Total probability of testing positive (P(Positive Test))
P_positive_test = (P_positive_given_disease * P_disease) + (P_positive_given_no_disease * P_no_disease)

# Posterior probability of having the disease given a positive test result (P(Disease | Positive Test))
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive_test

# Output the result
print("\nP(E) (Marginal probability of testing positive): ", P_positive_test)
print(f"\P(H1|E) (Probability of having the disease given a positive test result: {P_disease_given_positive:.2f}")
