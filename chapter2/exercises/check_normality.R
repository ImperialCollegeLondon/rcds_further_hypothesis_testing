# Load libraries
library(ggplot2)

# Set random seed
set.seed(123)

# Formulate hypotheses
# H0: Data follows a normal distribution
# H1: Data does not follow a normal distribution

# Generate random data sampling from Gaussian
sample_size <- 100
observed_data <- rnorm(sample_size, mean = 0, sd = 1)
# observed_data <- sample(0:9, size = sample_size, replace = TRUE)

# Generate points for the normal distribution
x <- seq(min(observed_data), max(observed_data), length.out = 500)
y <- dnorm(x, mean = mean(observed_data), sd = sd(observed_data))

# Define bins and calculate observed frequencies
num_bins <- 10
bin_edges <- seq(min(observed_data), max(observed_data), length.out = num_bins + 1)
hist_counts <- hist(observed_data, breaks = bin_edges, plot = FALSE)$counts

# Plot observations and Gaussian fit
df <- data.frame(x = observed_data)
ggplot(df, aes(x = x)) +
  geom_histogram(aes(y = ..density..), bins = num_bins, fill = "skyblue", alpha = 0.6, color = "black") +
  stat_function(fun = dnorm, args = list(mean = mean(observed_data), sd = sd(observed_data)),
                color = "red", size = 1) +
  ggtitle("Histogram of Observations") +
  xlab("Value") +
  ylab("Density") +
  theme_minimal()

# Compute expected frequencies
cdf_values <- pnorm(bin_edges, mean = mean(observed_data), sd = sd(observed_data))
expected_frequencies <- sample_size * diff(cdf_values)
# Normalize expected frequencies to match the sum of observed frequencies
expected_frequencies <- expected_frequencies * sum(hist_counts) / sum(expected_frequencies)

# Compute Chi-square statistic manually
chi_square_manual <- sum((hist_counts - expected_frequencies)^2 / expected_frequencies)
df <- num_bins - 1
p_value_manual <- 1 - pchisq(chi_square_manual, df = df)
cat("\nChi-square Statistic (manual):", chi_square_manual, "\n")
cat("P-value (manual):", p_value_manual, "\n")

# Compute Chi-square statistic using R's chisq.test
chisq_test <- chisq.test(hist_counts, p = expected_frequencies / sum(expected_frequencies), rescale.p = FALSE)
chi_square_scipy <- chisq_test$statistic
p_value <- chisq_test$p.value
cat("\nChi-square Statistic (chisq.test):", chi_square_scipy, "\n")
cat("P-value (chisq.test):", p_value, "\n")

# Interpret result
alpha <- 0.05
if (p_value < alpha) {
  cat("\nReject H0: The data does not follow a normal distribution.\n")
} else {
  cat("\nFail to reject H0: The data follows a normal distribution.\n")
}

# Plot the chi-square distribution
x <- seq(0, 30, length.out = 500)
y <- dchisq(x, df = df)

chi2_df <- data.frame(x = x, y = y)
ggplot(chi2_df, aes(x = x, y = y)) +
  geom_line(color = "blue") +
  geom_vline(xintercept = chi_square_manual, color = "red", linetype = "dashed") +
  annotate("text", x = chi_square_manual, y = max(y) / 2, 
           label = sprintf("Chi-square Observed: %.2f", chi_square_manual), hjust = -0.2) +
  ggtitle("Chi-square Goodness-of-Fit Test") +
  xlab("Chi-square Value") +
  ylab("Density") +
  theme_minimal()
