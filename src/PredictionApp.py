import pathlib
import tkinter

import customtkinter
from tkinter import filedialog

import pandas as pd
import seaborn as sns
import numpy as np

from tab_views.DatasetTabView import DatasetTabView
from tab_views.LRTabView import LRTabView
from tab_views.MLRTabView import MLRTabView
from tab_views.SVRTabView import SVRTabView
from tab_views.RFRTabView import RFRTabView
# from tab_views.NNRTabView import NNRTabView

def center(win, parent=None):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    :param parent: the parent window relative to which `win` will be centered
    """
    win.update_idletasks()
    win.withdraw()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()

    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width

    if parent is None:
        x = win.winfo_screenwidth() // 2 - win_width // 2
        y = win.winfo_screenheight() // 2 - win_height // 2
    else:
        x = parent.winfo_width() // 2 - win_width // 2 + parent.winfo_x()
        y = parent.winfo_height() // 2 - win_height // 2 + parent.winfo_y()

    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()


class PredictionApp(customtkinter.CTk):
    """
    This class contains the complete user interface of the app
    to compare multiple regression methods
    """
    def __init__(self):
        super().__init__()

        self.__dataset_path: str = ''
        self.__dataset = pd.DataFrame()
        self.__predictable_col_string_var = None

        self.__loading_widget = None
        self.__loading_widget_label: customtkinter.CTkLabel
        self.__progress_bar: customtkinter.CTkProgressBar

        self.__window_width = 950
        self.__window_height = 800

        self.title("Profit Prediction")
        self.geometry(f"{self.__window_width}x{self.__window_height}")

        center(self)

        self.__menu_bar: tkinter.Menu
        self.__create_menu_bar()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.__sidebar_frame = customtkinter.CTkFrame(self, width=100, corner_radius=10)
        self.__sidebar_frame.grid(row=0, column=0, rowspan=4, padx=(10, 0), pady=(10, 10), sticky="nsew")
        self.__sidebar_frame.grid_rowconfigure(5, weight=1)

        self.__import_button = customtkinter.CTkButton(self.__sidebar_frame, text='Import Dataset',
                                                       command=self.import_dataset)
        self.__import_button.grid(row=0, column=0, padx=20, pady=10)

        self.__predict_button = customtkinter.CTkButton(self.__sidebar_frame, text='Predict', state='disabled', command=self.__predict)
        self.__predict_button.grid(row=1, column=0, padx=20, pady=10)

        self.__accuracy_button = customtkinter.CTkButton(self.__sidebar_frame, text='Accuracy', state='disabled', command=self.__accuracy)
        self.__accuracy_button.grid(row=2, column=0, padx=20, pady=10)

        self.__predictable_col_label = customtkinter.CTkLabel(self.__sidebar_frame, text='To Predict:')
        self.__predictable_col_label.grid(row=3, column=0, padx=20)

        self.__predictable_column_option_menu = customtkinter.CTkOptionMenu(self.__sidebar_frame, values=[])
        self.__predictable_column_option_menu.set('N/A')
        self.__predictable_column_option_menu.grid(row=4, column=0, padx=20, pady=10)

        self.__reload_button = customtkinter.CTkButton(self.__sidebar_frame, text='Reload Dataset', state='disabled', command=self.__reload_dataset)
        self.__reload_button.grid(row=6, column=0, padx=20, pady=10)

        self.__dataset_name_label = customtkinter.CTkLabel(self.__sidebar_frame, text='No Dataset Loaded', wraplength=150,
                                                           font=customtkinter.CTkFont(weight='bold'))
        self.__dataset_name_label.grid(row=7, column=0, padx=20, pady=10)

        self.__main_tab_view = customtkinter.CTkTabview(self, corner_radius=10)
        self.__main_tab_view.grid(row=0, column=1, rowspan=8, columnspan=3, padx=(10, 10), pady=(10, 10), sticky="nsew")

        # self.__tab_view_types = [DatasetTabView, LRTabView, MLRTabView, SVRTabView, RFRTabView, NNRTabView]
        self.__tab_view_types = [DatasetTabView, LRTabView, MLRTabView, SVRTabView, RFRTabView]
        self.__tab_views = []

        for tab_view_type in self.__tab_view_types:
            tab_name = tab_view_type.get_tab_name()
            self.__main_tab_view.add(tab_name)
            self.__tab_views.append(tab_view_type(self.__main_tab_view.tab(tab_name)))

        sns.set_theme()

    def import_dataset(self):
        path = filedialog.askopenfilename(
            title='Select dataset file',
            initialdir=pathlib.Path(__file__).parent.parent,
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if path == '':
            return

        if self.__dataset_path != path:
            self.__dataset_path = path
            file_name = path.split('/')[-1]
            self.__dataset_name_label.configure(text=file_name)

            self.__create_loading_widget()
            self.__reload_dataset()
            self.__loading_widget.destroy()

    def __reload_dataset(self):
        if len(self.__dataset_path):
            self.__dataset = pd.read_csv(self.__dataset_path)
            self.__dataset = self.__dataset.select_dtypes(include=np.number)
            self.__dataset.dropna(inplace=True)

            attribute_list = list(self.__dataset.columns.values)

            self.__predict_button.configure(state='normal')
            self.__accuracy_button.configure(state='normal')
            self.__reload_button.configure(state='normal')

            self.__predictable_col_string_var = tkinter.StringVar()
            self.__predictable_col_string_var.set(attribute_list[-1])

            self.__predictable_column_option_menu.configure(values=attribute_list,
                                                            variable=self.__predictable_col_string_var)
            self.__predictable_col_string_var.trace("w", self.__predictable_column_callback)

            selected_col = self.__predictable_col_string_var.get()

            for tab_view in self.__tab_views:
                self.__loading_widget_label.configure(text=f'Creating {type(tab_view).__name__}....')
                self.__progress_bar.step()
                self.__loading_widget.update_idletasks()
                tab_view.invalidate(self.__dataset, selected_col)
        else:
            print('ERROR: Failed to reload dataset!')

    def __create_loading_widget(self):
        self.__loading_widget = customtkinter.CTkToplevel(self)
        self.__loading_widget.title('Loading Dataset')
        self.__loading_widget.geometry('400x75')
        center(self.__loading_widget, parent=self)

        self.__loading_widget.grab_set()
        self.__loading_widget.transient(self)

        self.__loading_widget_label = customtkinter.CTkLabel(self.__loading_widget, text='Loading dataset...', font=customtkinter.CTkFont(size=15, weight="bold"))
        self.__loading_widget_label.pack(pady=10)

        self.__progress_bar = customtkinter.CTkProgressBar(self.__loading_widget, mode='indeterminate', width=300)
        self.__progress_bar.pack()
        self.__progress_bar.start()

    def __predictable_column_callback(self, *args):
        column = self.__predictable_col_string_var.get()
        print(column)
        for tab_view in self.__tab_views:
            tab_view.invalidate(self.__dataset, column)

    def __predict(self):
        current_tab = self.__main_tab_view.get()
        for tab_view in self.__tab_views:
            if current_tab == type(tab_view).get_tab_name():
                tab_view.predict()

    def __accuracy(self):
        current_tab = self.__main_tab_view.get()
        for tab_view in self.__tab_views:
            if current_tab == type(tab_view).get_tab_name():
                tab_view.accuracy()

    def __create_menu_bar(self):
        self.__menu_bar = tkinter.Menu(self)
        self.configure(menu=self.__menu_bar)

        fileMenu = tkinter.Menu(self.__menu_bar)
        self.__menu_bar.add_cascade(label="File", menu=fileMenu)

        editMenu = tkinter.Menu(self.__menu_bar)
        self.__menu_bar.add_cascade(label="Edit", menu=editMenu)

        fileMenu.add_command(label="Item")
        fileMenu.add_command(label="Exit")
        editMenu.add_command(label="Undo")
        editMenu.add_command(label="Redo")