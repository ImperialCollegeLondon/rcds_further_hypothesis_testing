# RCDS Further hypothesis testing
# Jesús Urtasun Elizari - Imperial College London
# Chapter 1 - Comparing variances

# Chapter 1. Parameter estimation, comparing means and variances

# Install libraries
library(tidyverse)

# Set working directory
setwd("/Users/jurtasun/Desktop/Courses/ICL/RCDS_Further_hypothesis_testing/chapter1")

# Question: do types 4 and 5 have the same variance in luminosity?

# Read input data
data <- read_csv("stars.csv")
type_key <- c('Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant','Hypergiant')
spectral_classes <- c('O', 'B', 'A', 'F', 'G', 'K', 'M')

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
# H0; they come from same distributions, hence s1 ~ s2
# H1; they come from different distrinbutions, hence s1 != s2

# Get log luminosity and compute variances
table(data$type)
type4 <- data %>% filter(type == 4) %>% pull(luminosity) %>% log
type5 <- data %>% filter(type == 5) %>% pull(luminosity) %>% log
var4 <- var(type4)
var5 <- var(type5)
print(paste('Type 4:', var4))
print(paste('Type 5:', var5))
print(paste('difference:', var4 - var5))

# Variances are similar, but is the difference between them statistically significant?

# Compute F variable
fstat <- var(type4) / var(type5)
print(paste("F = ", fstat))

# Compute p-value
p_value <- pf(fstat, 39, 39)
# p_value <- pf(fstat, 39, 39) * 2
print(paste("p =", p_value))

# Plot distribution
x <- seq(0.1, 3, 0.01)
plot(x, df(x, 39, 39), xlab = "F", ylab = "pdf", type = "l", col = "grey")
x_region <- seq(0.1, fstat, 0.01)
polygon(c(x_region, fstat, 0.01), c(df(x_region, 39, 39), 0, 0), border = NA, col = "lightgrey")
