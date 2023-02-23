import customtkinter
from RegressionApp import RegressionApp

if __name__ == '__main__':
    customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

    app = RegressionApp()  # Possible Parameter: methods=(Regression.NeuralNetwork, Regression.MultipleLinear)
    app.mainloop()
