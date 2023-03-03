import customtkinter
from regression import Regression
from utils import center


class MethodSelectionWidget(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        width = 255
        height = 270
        self.title('Select Methods')
        self.geometry(f'{width}x{height}')

        center(self)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.__main_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.__main_frame.grid(row=0, column=0, sticky='NSEW', padx=10, pady=10)

        self.__main_frame.grid_columnconfigure(0, weight=1)

        self.__method_checkbox_variables = [True] * len(Regression)
        self.__method_checkboxes = []
        row_index = 0
        for method in Regression:
            self.__method_checkboxes.append(customtkinter.CTkCheckBox(self.__main_frame, text=method.to_string()))
            self.__method_checkboxes[-1].select()

        for checkbox in self.__method_checkboxes:
            checkbox.grid(row=row_index, column=0, padx=20, pady=(7, 7), sticky='NSEW')
            row_index += 1

        done_button = customtkinter.CTkButton(self.__main_frame, text='Done', width=200, command=self.__done_callback)
        done_button.grid(row=row_index, column=0, padx=20, pady=20, sticky='NSEW')

    def __done_callback(self, *args):
        i = 0
        for checkbox in self.__method_checkboxes:
            self.__method_checkbox_variables[i] = bool(checkbox.get())
            i += 1
        self.update_idletasks()
        self.destroy()

    def get_choices(self) -> list[bool]:
        return self.__method_checkbox_variables


if __name__ == '__main__':
    widget = MethodSelectionWidget()
    widget.mainloop()
    for method, var in zip(Regression, widget.get_choices()):
        print(method, var)
