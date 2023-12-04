import math

import customtkinter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DatasetTabView:
    """
    This class contains functions to describe a dataset using pandas module
    """
    __tab_name = 'Dataset'

    @staticmethod
    def get_tab_name() -> str:
        return DatasetTabView.__tab_name

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.__tab_view = tab_view
        self.__dataset: pd.DataFrame

        self.__tab_view.rowconfigure(0, weight=1)
        self.__tab_view.columnconfigure(0, weight=1)

        self.__inner_tab_view_names = ['Full Plot', 'Sub Plots']

        self.__inner_tab_view: customtkinter.CTkTabview
        self.__inner_tabs = []

    def __create_layout(self):
        assert not self.__dataset.empty
        self.__inner_tab_view = customtkinter.CTkTabview(self.__tab_view, height=10)
        self.__inner_tab_view.grid(row=0, column=0, sticky='NSEW')
        self.__inner_tabs = []

        for tab_name in self.__inner_tab_view_names:
            self.__inner_tab_view.add(tab_name)
            self.__inner_tabs.append(self.__inner_tab_view.tab(tab_name))
            self.__inner_tab_view.tab(tab_name).grid_rowconfigure(0, weight=1)
            self.__inner_tab_view.tab(tab_name).grid_columnconfigure(0, weight=1)

        self.__create_plot(plottype='full')
        self.__create_plot(plottype='subplot')

    def __create_plot(self, plottype: str):
        figure = plt.Figure()
        figure.set_layout_engine("constrained")
        ax = figure.subplots()

        if plottype == 'full':
            self.__dataset.plot(ax=ax)
        elif plottype == 'subplot':
            max_columns = 3
            length = len(self.__dataset.columns)
            if length < max_columns:
                layout = (1, length)
            else:
                rows = math.ceil(len(self.__dataset.columns) / max_columns)
                layout = (rows, max_columns)
            self.__dataset.plot(ax=ax, subplots=True, layout=layout)
        else:
            raise '`type` should be one of ["full", "subplot"]'

        ax.yaxis.set_label('Row Values')  # Fixme: Labels not being shown
        ax.xaxis.set_label('Row Indices')
        ax.yaxis.set_label_position("right")

        figure.tight_layout()

        if plottype == 'full':
            canvas = FigureCanvasTkAgg(figure, master=self.__inner_tab_view.tab(self.__inner_tab_view_names[0]))
        elif plottype == 'subplot':
            canvas = FigureCanvasTkAgg(figure, master=self.__inner_tab_view.tab(self.__inner_tab_view_names[1]))
        else:
            raise '`type` should be one of ["full", "subplot"]'

        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='NSEW')

    def __invalidate_widgets(self):
        for widgets in self.__tab_view.winfo_children():
            widgets.destroy()

    def invalidate(self, dataset: pd.DataFrame, *args):
        self.__dataset = dataset

        self.__invalidate_widgets()
        self.__create_layout()
