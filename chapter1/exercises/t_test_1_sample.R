# Import libraries
library(ggplot2)
library(MASS)

# Set random seed
set.seed(42)

# One sample t-test
cat("\nOne sample t-test:\nCompare sample mean to hypothesized value\n")

# Formulate hypotheses
# H0 (null hypothesis): The coin is fair (p = 0.5)
# H1 (alternative hypothesis): The coin is biased (p != 0.5)

# Simulate 100 coin tosses (1 = heads, 0 = tails)
observations <- sample(c(0, 1), size = 100, prob = c(0.5, 0.5), replace = TRUE)
# observations <- sample(c(0, 1), size = 100, prob = c(0.3, 0.7), replace = TRUE)

# Plot histogram of the observations
ggplot(data.frame(observations), aes(x = observations)) +
  geom_bar(stat = "count", fill = "skyblue", color = "black", alpha = 0.75) +
  scale_x_continuous(breaks = c(0, 1), labels = c("Tails (0)", "Heads (1)")) +
  labs(title = "Histogram of 100 coin tosses", x = "Outcome", y = "Frequency")

# Compute observed mean and standard deviation
n <- length(observations)
mean_observed <- mean(observations)
std_dev_observed <- sd(observations)  # Sample standard deviation

# Compute the t statistic and p-value (Expected mean under H0 = 0.5)
expected_mean <- 0.5
t_stat_manual <- (mean_observed - expected_mean) / (std_dev_observed / sqrt(n))
p_value_manual <- 2 * (1 - pt(abs(t_stat_manual), df = n - 1))
cat(sprintf("\nT statistic (manual): %.4f\n", t_stat_manual))
cat(sprintf("P-value (manual): %.4f\n", p_value_manual))

# Compute the t statistic using a library
t_test_result <- t.test(observations, mu = 0.5)
cat("\nUsing t.test() function:\n")
cat(sprintf("\nT statistic (library): %.4f\n", t_test_result$statistic))
cat(sprintf("P-value (library): %.4f\n", t_test_result$p.value))

# Parameters for plot t-distribution
x <- seq(-4, 4, length.out = 1000)  # Range for t-distribution
t_dist <- dt(x, df = n - 1)  # Probability density function for t-distribution

# Plot t-distribution
plot(x, t_dist, type = "l", col = "blue", lwd = 2, 
     xlab = "t value", ylab = "Density", main = "t-distribution and observed t statistic")
abline(v = t_stat_manual, col = "red", lty = 2)
abline(v = -t_stat_manual, col = "red", lty = 2)
legend("topright", legend = paste("t statistic =", round(t_stat_manual, 2)), 
       col = "red", lty = 2)

# Interpret result (significance level 0.05)
alpha <- 0.05
if (p_value_manual < alpha) {
  cat("Reject H0: The coin is likely biased.\n")
} else {
  cat("Accept H0: No evidence of biased coin.\n")
}
