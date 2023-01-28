import pandas as pd
import random
import numpy as np


# (y - y1) * (y - y1) = 4 * a * (x - x1)

def parabola(x, x1=0, y1=0):
    return y1 + (x - x1) * (x - x1) / 4


def generate_random_dataset(size=1000):
    x = []
    y = []
    ms_range = [100000, 500000]
    x_offset = np.floor((ms_range[1] - ms_range[0]) / size)

    new_origin_x = ms_range[0]
    new_origin_y = parabola(ms_range[0])

    for i in range(0, size):
        random_factor_x = int(ms_range[0] / 100)
        random_factor_y = int(random_factor_x * random_factor_x / 1000)

        random_x = random.randint(-random_factor_x, random_factor_x)
        random_y = random.randint(-random_factor_y, random_factor_y)

        # x_val = i + ms_range[0]
        x_plot = i * x_offset
        x_val = ms_range[0] + i * x_offset
        x.append(x_val)
        # y.append(parabola(x_val) / 100000 + random_y)
        y.append(parabola(x_plot))

    np.interp(y, [y[0], y[size - 1]], ms_range)
    dataset = pd.DataFrame({'Marketing Spend': x, 'Profit': y})
    return dataset
