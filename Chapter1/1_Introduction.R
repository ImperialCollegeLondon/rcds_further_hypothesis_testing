# Import packages
library(tidyverse)

# Set working directory
setwd("/Users/jurtasun/Desktop/Courses/ICL/RCDS_Statistics2/Chapter1")

set_plot_dimensions <- function(width_choice, height_choice) {
      options(repr.plot.width=width_choice, repr.plot.height=height_choice)
}

cbPal <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#0072B2", "#D55E00")
set_plot_dimensions(5, 4)

# Load input data
data <- read_csv("stars.csv")
type_key <- c('Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant','Hypergiant')
spectral_classes <- c('O','B','A','F','G','K','M')

# Process data
data$type <- factor(data$type)
data$spectral_class <- factor(data$spectral_class, levels=spectral_classes)
head(data)

# Plot
ggplot(data, aes(x=luminosity)) + 
      geom_histogram(bins=50, alpha=0.5)

# Plot log scale
ggplot(data, aes(x=log(luminosity))) + 
      geom_histogram(bins=50, alpha=0.5)

# Split by another variable to compare different groups
ggplot(data, aes(x=type, y=log(luminosity), fill=type)) + 
      scale_fill_manual(values=cbPal) +
      geom_boxplot(alpha=0.5) + 
      guides(fill="none") +
      coord_flip()       
