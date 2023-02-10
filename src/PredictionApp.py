import pathlib
import tkinter

import customtkinter
from tkinter import filedialog

import pandas as pd
import seaborn as sns
import numpy as np

from src.tab_views.DatasetTabView import DatasetTabView
from src.tab_views.LRTabView import LRTabView
from src.tab_views.MLRTabView import MLRTabView
from src.tab_views.SVRTabView import SVRTabView


class PredictionApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.__dataset_path: str = ''
        self.__dataset = pd.DataFrame()
        self.__predictable_col_string_var = None

        self.title("Profit Prediction")
        self.geometry(f"{950}x{800}")

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

        self.__main_tab_view.add(DatasetTabView.get_tab_name())
        self.__main_tab_view.add(LRTabView.get_tab_name())
        self.__main_tab_view.add(MLRTabView.get_tab_name())
        self.__main_tab_view.add(SVRTabView.get_tab_name())

        self.__DatasetTabView = DatasetTabView(self.__main_tab_view.tab(DatasetTabView.get_tab_name()))
        self.__LRTabView = LRTabView(self.__main_tab_view.tab(LRTabView.get_tab_name()))
        self.__MLRTabView = MLRTabView(self.__main_tab_view.tab(MLRTabView.get_tab_name()))
        self.__SVRTabView = SVRTabView(self.__main_tab_view.tab(SVRTabView.get_tab_name()))

        sns.set_theme()

    def import_dataset(self):
        path = filedialog.askopenfilename(
            title='Select dataset file',
            initialdir=pathlib.Path(__file__).parent,
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if path == '':
            return

        if self.__dataset_path != path:
            self.__dataset_path = path
            file_name = path.split('/')[-1]
            self.__dataset_name_label.configure(text=file_name)

            self.__reload_dataset()

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
            self.__DatasetTabView.invalidate(self.__dataset)
            self.__LRTabView.invalidate(self.__dataset, selected_col)
            self.__MLRTabView.invalidate(self.__dataset, selected_col)
            self.__SVRTabView.invalidate(self.__dataset, selected_col)
        else:
            print('ERROR: Failed to reload dataset!')

    def __predictable_column_callback(self, *args):
        column = self.__predictable_col_string_var.get()
        print(column)
        self.__LRTabView.set_predictable_column(column)
        self.__MLRTabView.invalidate(self.__dataset, column)
        self.__SVRTabView.invalidate(self.__dataset, column)

    def __predict(self):
        current_tab = self.__main_tab_view.get()
        if current_tab == self.__LRTabView.get_tab_name():
            self.__LRTabView.predict()
        elif current_tab == self.__MLRTabView.get_tab_name():
            self.__MLRTabView.predict()
        elif current_tab == self.__SVRTabView.get_tab_name():
            self.__SVRTabView.predict()

    def __accuracy(self):
        current_tab = self.__main_tab_view.get()
        if current_tab == self.__LRTabView.get_tab_name():
            self.__LRTabView.accuracy()
        elif current_tab == self.__MLRTabView.get_tab_name():
            self.__MLRTabView.accuracy()
        elif current_tab == self.__SVRTabView.get_tab_name():
            self.__SVRTabView.accuracy()
