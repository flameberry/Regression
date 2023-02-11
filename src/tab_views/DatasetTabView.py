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
        self.__dataset = pd.DataFrame()

        self.__tab_view.rowconfigure(0, weight=1)
        self.__tab_view.columnconfigure(0, weight=1)

    def __create_layout(self):
        assert not self.__dataset.empty
        figure = plt.Figure()
        figure.set_layout_engine("constrained")
        ax = figure.subplots()

        self.__dataset.plot(ax=ax)

        ax.yaxis.set_label('Row Values')  # Fixme: Labels not being shown
        ax.xaxis.set_label('Row Indices')
        ax.yaxis.set_label_position("right")

        canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='NSEW')

    def __invalidate_widgets(self):
        for widgets in self.__tab_view.winfo_children():
            widgets.destroy()

    def invalidate(self, dataset: pd.DataFrame, *args):
        self.__dataset = dataset

        self.__invalidate_widgets()
        self.__create_layout()
