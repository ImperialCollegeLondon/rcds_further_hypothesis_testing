# Load libraries
library(ggplot2)

# Simulate random choices
simulate_random_choices <- function(n_people) {
  # Uniform distribution
  random_choices <- sample(0:9, size = n_people * 3, replace = TRUE)
  return(random_choices)
}

# Simulate human-biased number choices
simulate_human_choices <- function(n_people) {
  # Frequency vector for numbers 1 to 9
  frequencies <- c(1, 2, 1, 2, 1, 2, 1, 2, 1)
  total_frequency <- sum(frequencies)
  
  # Normalize frequencies / compute probabilities
  human_bias <- frequencies / total_frequency
  
  # Add 0 with small frequency to complete bias for numbers 0-9
  human_bias <- c(1 / total_frequency, human_bias)
  
  # Check sum of probabilities equals 1
  human_bias <- human_bias / sum(human_bias)
  
  # Simulate choices using the computed bias (each person picks 3 numbers)
  choices <- sample(0:9, size = n_people * 3, replace = TRUE, prob = human_bias)
  return(choices)
}

# Set random seed
set.seed(123)

# Compare expected vs observed choices
cat("\nCompare expected vs observed choices\n")

# Formulate hypotheses
# H0: Data follows a uniform distribution
# H1: Significant difference from uniform distribution

# Prepare expected and observed choices
n_people <- 10 # Number of participants
human_choices <- simulate_human_choices(n_people)
random_choices <- simulate_random_choices(n_people)

# Count frequencies
random_freq <- table(factor(random_choices, levels = 0:9))
human_freq <- table(factor(human_choices, levels = 0:9))

# Perform chi-square test
expected_freq <- rep(n_people * 3 / 10, 10)  # Expected frequency for uniform distribution
chi_square_test <- chisq.test(as.numeric(human_freq), p = expected_freq / sum(expected_freq), rescale.p = FALSE)

# Plot the results
df <- data.frame(
  Numbers = factor(0:9),
  Human_Choices = as.numeric(human_freq),
  Random_Choices = as.numeric(random_freq)
)

ggplot(df, aes(x = Numbers)) +
  geom_bar(aes(y = Human_Choices), stat = "identity", fill = "skyblue", width = 0.4, position = position_nudge(x = -0.2)) +
  geom_bar(aes(y = Random_Choices), stat = "identity", fill = "orange", width = 0.4, position = position_nudge(x = 0.2)) +
  geom_hline(yintercept = n_people * 3 / 10, color = "gray", linetype = "dashed") +
  labs(
    title = "Distribution of Numbers: Human Choices vs Random Choices",
    x = "Numbers",
    y = "Frequency"
  ) +
  theme_minimal() +
  scale_y_continuous(expand = c(0, 0)) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  guides(fill = guide_legend(title = NULL))

# Print results
cat("Chi-Square Test Results:\n")
cat(sprintf("Chi-Square Statistic: %.2f\n", chi_square_test$statistic))
cat(sprintf("P-value: %.4f\n", chi_square_test$p.value))

if (chi_square_test$p.value < 0.05) {
  cat("\nThe observed distribution significantly deviates from a uniform distribution.\n")
} else {
  cat("\nNo significant deviation from a uniform distribution.\n")
}
