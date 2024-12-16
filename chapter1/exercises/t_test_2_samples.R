# Import libraries
library(ggplot2)

# Set random seed
set.seed(0)

# Two sample t-test
cat("\nTwo sample t-test:\nCompare sample means of two independent groups\n")

# Formulate hypotheses
# H0: x1, x2 come from the same distribution (mean1 = mean2 = mu)
# H1: x1, x2 come from different distributions (mean1 != mean2 != mu)

# Simulate two sets of Gaussian observations
sample1 <- rnorm(30, mean = 50, sd = 5)
sample2 <- rnorm(30, mean = 55, sd = 5)

# Plot observations as histograms
data <- data.frame(value = c(sample1, sample2), group = rep(c("Sample 1", "Sample 2"), each = 30))
ggplot(data, aes(x = value, fill = group)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 15, color = "black") +
  labs(title = "Histogram of Gaussian Observations", x = "Value", y = "Frequency") +
  scale_fill_manual(values = c("blue", "red"))

# Compute the sample means and standard deviations
mean1 <- mean(sample1)
mean2 <- mean(sample2)
std1 <- sd(sample1)
std2 <- sd(sample2)
n1 <- length(sample1)
n2 <- length(sample2)

# Compute standard error of the difference
se <- sqrt((std1^2 / n1) + (std2^2 / n2))

# Compute the t-statistic
t_stat_manual <- (mean1 - mean2) / se
p_value_manual <- 2 * (1 - pt(abs(t_stat_manual), df = n1 + n2 - 2))
cat(sprintf("\nManual calculation:\nt-statistic = %.4f\n", t_stat_manual))
cat(sprintf("p-value = %.4f\n", p_value_manual))

# Compute two-tailed statistic and p-value using t.test()
t_test_result <- t.test(sample1, sample2)
cat("\nUsing t.test() function:\n")
cat(sprintf("\nt-statistic = %.4f\n", t_test_result$statistic))
cat(sprintf("p-value = %.4f\n", t_test_result$p.value))

# Parameters for plot t-distribution
x <- seq(-4, 4, length.out = 1000)  # Range for t-distribution
t_dist <- dt(x, df = n1 + n2 - 2)  # Probability density function for t-distribution

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
  cat("Reject H0: Samples come from different distributions.\n")
} else {
  cat("Accept H0: Samples come from same distribution.\n")
}
