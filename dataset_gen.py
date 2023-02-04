import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a * np.power(abs(x), b)


def power_law_reverse(y, a, b):
    return pow(abs(y) / a, 1 / b)


def log_reverse(y, a, b):
    e = 2.718281828459045
    base = 1.69
    return base ** (y / a) / b


def bell_curve(x, a, b, c):
    # e = 2.718281828459045
    e = 4
    return a * e ** (((x - b) ** 2) / (2 * c * c))


def bell_curve_reverse(y, a, b, c):
    return np.sqrt(abs(2 * c * c * np.log(abs(y / a)))) + b


def quadratic(x, a, b, c):
    return a * x * x + b * x + c


def quadratic_reverse(y, a, b, c):
    delta = b * b - 4 * a * (c - y)
    if hasattr(delta, "__len__"):
        for i in range(len(delta)):
            if delta[i] < 0:
                delta[i] = 0

    return (-b + np.sqrt(delta)) / (2 * a)


def cubic_reverse(y, a, b):
    return pow((y - b) / a, 1 / 3)


def sine(x, a, b):
    return a * np.sin(b * x)


def sine_reverse(y, a, b):
    return np.arcsin(y / a) / b


def generate_random_dataset(size=10000):
    range_min = 50000
    range_max = 650000

    mspend_dummy = np.linspace(start=range_min, stop=range_max, num=size)
    noise_mspend = 0.3 * np.random.normal(loc=mspend_dummy.mean(), scale=mspend_dummy.std(), size=mspend_dummy.size)
    mspend_dummy += noise_mspend

    # y_dummy = np.array(power_law(mspend_dummy, 6500, 0.3))
    profit_dummy = np.array(sine(mspend_dummy, 1000000, 1.57 / mspend_dummy.max()))
    noise = 0.2 * np.random.normal(loc=profit_dummy.mean(), scale=profit_dummy.std(), size=profit_dummy.size)  # Add noise from a Gaussian distribution
    profit_dummy += noise

    admin_dummy = 10 * np.array(bell_curve_reverse(profit_dummy, 10 ** 5, 10 ** 2, 10 ** 4.5))
    noise_admin = 0.2 * np.random.normal(loc=admin_dummy.mean(), scale=admin_dummy.std(), size=admin_dummy.size)
    admin_dummy += noise_admin

    # rdspend_dummy = 40000 * np.array(sine_reverse(y_dummy, y_dummy.max(), 0.1))
    rdspend_dummy = np.array(power_law_reverse(profit_dummy + 2 * 10 ** 5, 7500, 0.375))
    noise_rdspend = 0.2 * np.random.normal(loc=rdspend_dummy.mean(), scale=rdspend_dummy.std(), size=rdspend_dummy.size)
    rdspend_dummy += noise_rdspend

    print('Marketing Spend Noise Range:', noise_mspend.min(), noise_mspend.max())
    print('Admin Noise Range:', noise_admin.min(), noise_admin.max())
    print('RDSpend Noise Range:', noise_rdspend.min(), noise_rdspend.max())
    print('Profit Noise Range:', noise.min(), noise.max())

    print('')

    print(f'Marketing Spend range: {mspend_dummy.min()}, {mspend_dummy.max()}')
    print(f'Admin range: {admin_dummy.min()}, {admin_dummy.max()}')
    print(f'RDSpend range: {rdspend_dummy.min()}, {rdspend_dummy.max()}')
    print(f'Profit range: {profit_dummy.min()}, {profit_dummy.max()}')

    dataset = pd.DataFrame({'Marketing Spend': mspend_dummy, 'Administration': admin_dummy, 'R&DSpend': rdspend_dummy, 'Profit': profit_dummy})
    return dataset


if __name__ == '__main__':
    random_dataset = generate_random_dataset()
    random_dataset.to_csv('datasets/random_dataset.csv', index=False)
    random_dataset.plot()
    plt.show()
