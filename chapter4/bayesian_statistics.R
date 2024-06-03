# Import libraries
library(MCMCpack)

# Set working directory
workdir <- "/Users/jurtasun/Desktop/Courses/ICL/RCDS_Further_hypothesis_testing/chapter4"
setwd(workdir)

# Bayesian Inference: Updating beliefs about a coin's bias
bayesian_inference <- function(data, prior_params) {
      posterior_params <- prior_params + c(sum(data == 1), sum(data == 0))
      return(posterior_params)
}

# Bayesian Modeling: Gaussian data fitting with known variance
bayesian_modeling <- function(data, prior_params) {
      prior_mean <- prior_params[1]
      prior_std <- prior_params[2]
      likelihood_mean <- mean(data)
      likelihood_std <- sd(data)
      posterior_mean <- (prior_mean / prior_std^2 + sum(data) / likelihood_std^2) / (1 / prior_std^2 + length(data) / likelihood_std^2)
      posterior_std <- sqrt(1 / (1 / prior_std^2 + length(data) / likelihood_std^2))
      return(c(posterior_mean, posterior_std))
}

# Example data
coin_flips <- rbinom(100, 1, 0.7)  # 1 for heads, 0 for tails
gaussian_data <- rnorm(100, mean = 5, sd = 2)

# Bayesian Inference example
prior_params <- c(1, 1)  # Beta(1, 1) prior
posterior_params <- bayesian_inference(coin_flips, prior_params)
print(paste("Bayesian Inference: Posterior parameters (alpha, beta):", posterior_params))

# Bayesian Modeling example
prior_params <- c(0, 1)  # Gaussian prior with mean=0, sd=1
posterior_params <- bayesian_modeling(gaussian_data, prior_params)
print(paste("Bayesian Modeling: Posterior mean and sd:", posterior_params))

# Bayesian Computation example (using MCMCpack for Metropolis-Hastings)
# posterior_samples <- MCMCmetrop1R(dnorm(gaussian_data), theta.init = mean(gaussian_data), data = gaussian_data, n.iter = 1000)
# plot(posterior_samples)
