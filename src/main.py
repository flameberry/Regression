import customtkinter
from PredictionApp import PredictionApp

if __name__ == '__main__':
    customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

    # customtkinter.set_widget_scaling(1.6)

    app = PredictionApp()
    app.mainloop()
