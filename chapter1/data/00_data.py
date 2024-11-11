# Import libraries
import csv
import random

# Number of observations
num_observations = 100

# Ranges for generating random values
luminosity_range = (0.001, 100)  # arbitrary units, from 0.001 to 100 times the Sun's luminosity
temperature_range = (2000, 40000)  # temperatures in Kelvin, typical for various star types
size_range = (0.1, 10)  # relative size, from 0.1 to 10 times the Sun's radius

# Define spectral class boundaries by temperature in Kelvin
spectral_classes = [
    ('O', 30000, 40000),  # O-type stars (hottest)
    ('B', 10000, 30000),  # B-type stars
    ('A', 7500, 10000),   # A-type stars
    ('F', 6000, 7500),    # F-type stars
    ('G', 5200, 6000),    # G-type stars (like the Sun)
    ('K', 3700, 5200),    # K-type stars
    ('M', 2000, 3700)     # M-type stars (coolest)
]

# Generate data
data = []
# Sample from uniform distribution
for _ in range(num_observations):
    luminosity = round(random.uniform(*luminosity_range), 3)
    temperature = round(random.uniform(*temperature_range), 2)
    size = round(random.uniform(*size_range), 3)
    
    # Determine the spectral class based on temperature
    for spectral_class, temp_min, temp_max in spectral_classes:
        if temp_min <= temperature <= temp_max:
            star_class = spectral_class
            break
    
    # Append generated data
    data.append([star_class, luminosity, temperature, size])

# Set output file name
csv_file = 'data_stars.csv'

# Write data to CSV
with open(csv_file, mode = 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Luminosity', 'Temperature', 'Size'])
    writer.writerows(data)

print(f"{csv_file} has been created with {num_observations} observations.")