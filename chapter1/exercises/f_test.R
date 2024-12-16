# Import libraries
library(ggplot2)

# Set random seed
set.seed(42)

# Fisher test
cat("\nFisher test:\nCompare sample variances of two independent groups\n")

# Simulate two sets of Gaussian observations
sample1 <- rnorm(30, mean = 50, sd = 5)
sample2 <- rnorm(30, mean = 55, sd = 7)

# Plot observations as histograms
data <- data.frame(value = c(sample1, sample2), group = rep(c("Sample 1", "Sample 2"), each = 30))
ggplot(data, aes(x = value, fill = group)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 15, color = "black") +
  labs(title = "Histogram of Gaussian Observations", x = "Value", y = "Frequency") +
  scale_fill_manual(values = c("blue", "red"))

# Calculate variances of the two samples
var1 <- var(sample1)
var2 <- var(sample2)

# Degrees of freedom for each sample
df1 <- length(sample1) - 1
df2 <- length(sample2) - 1

# Calculate F-statistic and p-value manually
F_stat_manual <- ifelse(var1 > var2, var1 / var2, var2 / var1)
p_value_manual <- 2 * (1 - pf(F_stat_manual, df1, df2))
cat(sprintf("\nManual calculation:\nF-statistic = %.4f\n", F_stat_manual))
cat(sprintf("p-value = %.4f\n", p_value_manual))

# Plot the F-distribution
x <- seq(0, 5, length.out = 1000)  # Range of F-values for plotting
f_dist <- df(x, df1, df2)  # F-distribution density function

# Plot F-distribution
plot(x, f_dist, type = "l", col = "blue", lwd = 2, 
     xlab = "F value", ylab = "Density", main = "F-distribution and observed F-statistic")
abline(v = F_stat_manual, col = "red", lty = 2)
legend("topright", legend = paste("Observed F =", round(F_stat_manual, 2)), 
       col = "red", lty = 2)

# Interpret result (significance level 0.05)
alpha <- 0.05
if (p_value_manual < alpha) {
  cat("Reject H0: Samples come from different distributions.\n")
} else {
  cat("Accept H0: Samples come from the same distribution.\n")
}
