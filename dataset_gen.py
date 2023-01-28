import pandas as pd
import numpy as np


# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a * np.power(x, b)


def generate_random_dataset(size=1000):
    range_min = 50000
    range_max = 500000

    x_dummy = np.linspace(start=range_min, stop=range_max, num=size)
    noise_x = 0.1 * np.random.normal(scale=x_dummy.std(), size=x_dummy.size)  # Fixme: Add standard deviation and mean
    x_dummy += noise_x

    y_dummy = np.array(power_law(x_dummy, 6500, 0.3))  # Add noise from a Gaussian distribution
    noise = 0.2 * np.random.normal(scale=y_dummy.std(), size=y_dummy.size)  # Fixme: Add standard deviation and mean
    y_dummy += noise

    dataset = pd.DataFrame({'Marketing Spend': x_dummy, 'Profit': y_dummy})
    return dataset
