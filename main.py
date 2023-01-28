import customtkinter
from PredictionApp import PredictionApp

import matplotlib.pyplot as plt
import dataset_gen

if __name__ == '__main__':
    random_dataset = dataset_gen.generate_random_dataset()
    random_dataset.to_csv('datasets/random_dataset.csv', index=False)

    # random_dataset.plot()
    # plt.show()

    customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    app = PredictionApp()
    app.mainloop()
