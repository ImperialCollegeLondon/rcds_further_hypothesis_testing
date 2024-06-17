# RCDS Further hypothesis testing
# Jesús Urtasun Elizari - Imperial College London
# Chapter 1 - Comparing means

# Chapter 1. Parameter estimation, comparing means and variances

# Install libraries
library(tidyverse)

# Set working directory
setwd("/Users/jurtasun/Desktop/Courses/ICL/RCDS_Further_hypothesis_testing/chapter1")

# Question: do types 4 and 5 have the same mean luminosity?
# Question: do types 2 and 3 have the same mean luminosity?

# Read input data
data <- read_csv("stars.csv")
type_key <- c('Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant','Hypergiant')
spectral_classes <- c('O', 'B', 'A', 'F', 'G', 'K', 'M')
      
# Explore data
head(data)
class(data)
dim(data)

# Store type and spectral class as factors
data$type <- factor(data$type)
data$spectral_class <- factor(data$spectral_class, levels=spectral_classes)

# Plot type 4 (supergiant) and type 5 (hypergiant)
data %>% filter(type %in% c(4,5)) %>%
      ggplot(aes(x = log(luminosity), fill = type)) + 
      geom_histogram(alpha = 0.5, bins = 10) + 
      guides(fill = "none") +
      theme_classic() + 
      facet_wrap(~ type, ncol = 1)

# Formulate null and alternative hypothesis
# H0: they come from same distribution
# H1: they come from different distributions

# Get log luminosity and compute mean values
table(data$type)
type4 <- data %>% filter(type == 4) %>% pull(luminosity) %>% log
type5 <- data %>% filter(type == 5) %>% pull(luminosity) %>% log
mean4 <- mean(type4)
mean5 <- mean(type5)
print(paste('Type 4:', mean4))
print(paste('Type 5:', mean5))
print(paste('difference:', mean4 - mean5))

# Mean values are similar, but is the difference between them statistically significant?
      
# Compute t-test
t.test(type4, type5, var.equal = TRUE, paired = FALSE)

# Store obtained t value and d.o.f
t_obs <- t.test(type4, type5, var.equal = TRUE, paired = FALSE)$statistic
df <- length(type4) + length(type5) - 2
print(paste("degrees of freedom:", df))

# Build range for t distribution
tmin <- -4
tmax <- 4
x <- seq(tmin, tmax, 0.01)
plot(x, dt(x, df), xlab = "t", ylab = "pdf", type = "l", col = "grey")

# The area of the shaded region is the two-tailed p-value
lower_tail <- seq(tmin, -t_obs, 0.01)
upper_tail <- seq(t_obs, tmax, 0.01)
polygon(c(lower_tail, -t_obs, tmin), c(dt(lower_tail, df), 0, 0), border = NA, col = "lightgrey")
polygon(c(upper_tail, tmax, t_obs), c(dt(upper_tail, df), 0, 0), border = NA, col = "lightgrey")

