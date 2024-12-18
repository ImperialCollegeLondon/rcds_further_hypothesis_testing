# Import libraries
library(ggplot2)
library(gridExtra)

# Set random seed
set.seed(42)

# Generate gaussian observations
group1 <- rnorm(10, mean = 5, sd = 1)
group2 <- rnorm(10, mean = 5, sd = 1)
group3 <- rnorm(10, mean = 5, sd = 1)
data <- list(group1, group2, group3)

# Plot observations
df <- data.frame(
  value = c(group1, group2, group3),
  group = factor(rep(1:3, each = 10))
)

p <- ggplot(df, aes(x = group, y = value)) +
  geom_violin(trim = FALSE, fill = "lightblue") +
  stat_summary(fun = mean, geom = "point", shape = 4, size = 3, color = "red") +
  labs(title = "Plot of observations", y = "Sample mean") +
  theme_minimal()

print(p)

# Compute group means
mean1 <- mean(group1)
mean2 <- mean(group2)
mean3 <- mean(group3)
overall_mean <- mean(c(group1, group2, group3))

# Compute sum of squares between (SSB) and within (SSW)
ssb <- length(data) * ((mean1 - overall_mean)^2 + (mean2 - overall_mean)^2 + (mean3 - overall_mean)^2)
ssw <- sum((group1 - mean1)^2) + sum((group2 - mean2)^2) + sum((group3 - mean3)^2)
df_between <- length(data) - 1  # n - 1
df_within <- length(group1) + length(group2) + length(group3) - length(data)  # N - n

# Mean Squares
msb <- ssb / df_between
msw <- ssw / df_within
df1 <- length(data)
df2 <- length(group1) + length(group2) + length(group3) - length(data)

# F-statistic
F_stat_manual <- msb / msw
p_value_manual <- 1 - pf(F_stat_manual, df1, df2)

cat("\nANOVA (manual):\n")
cat(sprintf("F statistic (manual): %.2f\n", F_stat_manual))
cat(sprintf("p value (manual): %.4f\n", p_value_manual))

# Compute ANOVA (built-in function)
anova_res <- aov(value ~ group, data = df)
summary_res <- summary(anova_res)
F_stat_scipy <- summary_res[[1]]$`F value`[1]
p_value_scipy <- summary_res[[1]]$`Pr(>F)`[1]

cat("\nANOVA (R built-in):\n")
cat(sprintf("F statistic (R built-in): %.2f\n", F_stat_scipy))
cat(sprintf("p value (R built-in): %.4f\n", p_value_scipy))

# Conclusion based on the p-value
alpha <- 0.05
if (p_value_manual < alpha) {
  cat("\nReject the null hypothesis: There are significant differences between the groups.\n")
} else {
  cat("\nFail to reject the null hypothesis: No significant differences between the groups.\n")
}
