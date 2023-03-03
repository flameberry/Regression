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


def ellipse_reverse(y, a, b):
    return np.sqrt(abs(a * a * (1 - (y * y) / (b * b))))


def half_life_inverse(y, a, offset_x=0, offset_y=0):
    return offset_x + a / (y - offset_y)


def rect_hyperbola_inverse(y, a):
    return np.sqrt(abs(a * a + y * y))


def generate_random_dataset(size=20000):
    range_min = 50000
    range_max = 650000

    mspend_dummy = np.linspace(start=range_min, stop=range_max, num=size)
    noise_mspend = 0.3 * np.random.normal(loc=mspend_dummy.mean(), scale=mspend_dummy.std(), size=mspend_dummy.size)
    mspend_dummy += noise_mspend
    mspend_dummy = abs(mspend_dummy)

    profit_dummy = np.array(sine(mspend_dummy, 1000000, 1.57 / mspend_dummy.max()))
    noise_profit = 0.5 * np.random.normal(loc=profit_dummy.mean(), scale=profit_dummy.std(), size=profit_dummy.size)
    profit_dummy += noise_profit
    profit_dummy = abs(profit_dummy)

    admin_dummy = 10 * np.array(bell_curve_reverse(profit_dummy, 10 ** 5, 10 ** 2, 10 ** 4.5))
    noise_admin = 0.9 * np.random.normal(loc=admin_dummy.mean(), scale=admin_dummy.std(), size=admin_dummy.size)
    admin_dummy += noise_admin
    admin_dummy = abs(admin_dummy)

    rdspend_dummy = np.array(power_law_reverse(profit_dummy + 0 * 10 ** 5, 7500, 0.375))
    noise_rdspend = 0.9 * np.random.normal(loc=rdspend_dummy.mean(), scale=rdspend_dummy.std(), size=rdspend_dummy.size)
    rdspend_dummy += noise_rdspend
    rdspend_dummy = abs(rdspend_dummy)

    sales_dummy = 10 ** 5 * np.array(cubic_reverse(profit_dummy, 4000, 1))
    noise_sales = 0.9 * np.random.normal(loc=sales_dummy.mean(), scale=sales_dummy.std(), size=sales_dummy.size)
    sales_dummy += noise_sales
    sales_dummy = abs(sales_dummy)

    # operations_dummy = np.array(ellipse_reverse(profit_dummy, 10 * 10 ** 5, 6 * 10 ** 5))
    # operations_dummy = np.array(rect_hyperbola_inverse(profit_dummy, 10 ** 6))
    operations_dummy = np.array(half_life_inverse(profit_dummy, 10 ** 11.7, 10 ** 6, -10 ** 5))
    noise_operations = 0.8 * np.random.normal(scale=operations_dummy.std(), size=operations_dummy.size)
    operations_dummy += noise_operations - 7 * 10 ** 5
    # operations_dummy = abs(operations_dummy)

    np.clip(operations_dummy, 10 ** 4, 20 * 10 ** 5)

    print('Marketing Spend Noise Range:', noise_mspend.min(), noise_mspend.max())
    print('Admin Noise Range:', noise_admin.min(), noise_admin.max())
    print('RDSpend Noise Range:', noise_rdspend.min(), noise_rdspend.max())
    print('Sales Noise Range:', noise_sales.min(), noise_sales.max())
    print('Operations Noise Range:', noise_operations.min(), noise_operations.max())
    print('Profit Noise Range:', noise_profit.min(), noise_profit.max())

    print('')

    print(f'Marketing Spend range: {mspend_dummy.min()}, {mspend_dummy.max()}')
    print(f'Admin range: {admin_dummy.min()}, {admin_dummy.max()}')
    print(f'RDSpend range: {rdspend_dummy.min()}, {rdspend_dummy.max()}')
    print(f'Sales range: {sales_dummy.min()}, {sales_dummy.max()}')
    print(f'Operations range: {operations_dummy.min()}, {operations_dummy.max()}')
    print(f'Profit range: {profit_dummy.min()}, {profit_dummy.max()}')

    dataset = pd.DataFrame({
        'Marketing Spend': mspend_dummy,
        'Administration': admin_dummy,
        'R&D Spend': rdspend_dummy,
        'Sales': sales_dummy,
        'Operations': operations_dummy,
        'Profit': profit_dummy
    })
    return dataset


if __name__ == '__main__':
    random_dataset = generate_random_dataset()
    random_dataset.to_csv('../datasets/random_dataset.csv', index=False)
    random_dataset.plot()
    plt.show()
