import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    dataset = pd.read_csv('../datasets/random_dataset_20k_noise_1p0.csv')
    ax = dataset.plot(subplots=True, layout=(3, 2))
    print(dataset.info())
    print(dataset.corr())
    plt.show()
