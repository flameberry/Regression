import customtkinter
from PredictionApp import PredictionApp

if __name__ == '__main__':
    customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    app = PredictionApp()
    app.mainloop()
