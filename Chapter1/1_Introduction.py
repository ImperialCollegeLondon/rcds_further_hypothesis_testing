# Import libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Load input data with Pandas
data = pd.read_csv("stars.csv")
type_key = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant','Hypergiant']
data.head()

# Get luminosity
sample = data.luminosity
xlab = 'luminosity'
ylab = 'freq'

# Plot
bins = np.linspace(sample.min(), sample.max())
ax = plt.axes()
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
plt.hist(sample, bins, alpha=0.5, color='gray')
plt.show()

# Apply log normalization
sample = data.luminosity.apply(np.log)
xlab = 'log(luminosity)'
ylab = 'freq'

# Plot
bins = np.linspace(sample.min(), sample.max())
ax = plt.axes()
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
plt.hist(sample, bins, alpha=0.5, color='gray')
plt.show()

# Split by another variable
sample = data.luminosity.apply(np.log)
grouped = sample.groupby(data.type)
xlab = 'log(luminosity)'
ylab = 'freq'

# Plot
bins = np.linspace(sample.min(), sample.max())
ax = plt.axes()
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
grouped.apply(lambda x: plt.hist(x, bins, alpha=0.5, color = 'C' + str(x.name), label=type_key[x.name]))
ax.legend(loc='upper center')
plt.show()