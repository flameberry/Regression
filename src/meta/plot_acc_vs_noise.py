import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
import pandas as pd
import seaborn as sns


def get_interpolated_x_y(x, y):
    X_Y_Spline = interp1d(x, y, kind='cubic')

    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    return X_, Y_


if __name__ == '__main__':
    accuracies_mlr = [99.99, 98.95, 94.44, 82.49, 67.81, 53.07, 43.17, 31.20, 18.61, 12.80, 6.07, 4.22, 1.15]
    accuracies_svr = [99.45, 99.20, 94.52, 82.33, 67.03, 51.88, 42.31, 28.96, 17.28, 11.62, 4.56, 1.35, 0.0]
    accuracies_rfr = [99.95, 99.04, 94.09, 79.79, 62.59, 48.86, 38.07, 27.05, 16.66, 11.28, 5.09, 3.53, 1.02]
    noise_factors = [0.0, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15]

    comp_acc = pd.DataFrame({
        'Noise': noise_factors,
        'MLR': accuracies_mlr,
        'SVR': accuracies_svr,
        'RFR': accuracies_rfr
    })

    sns.lineplot(data=comp_acc[['MLR', 'SVR', 'RFR']])

    plt.xlabel('Noise')
    plt.ylabel('Accuracy')
    plt.show()
