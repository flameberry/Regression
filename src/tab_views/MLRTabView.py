import customtkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from src.regression import Regression


class MLRTabView:
    """
    This class contains a Linear Regressor from sklearn to train on a dataset
    and predict values based on multiple columns
    """
    __tab_name = 'MLR'

    @staticmethod
    def get_tab_name() -> str:
        return MLRTabView.__tab_name

    @staticmethod
    def get_regression_type() -> Regression:
        return Regression.MultipleLinear

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.__dataset: pd.DataFrame
        self.__regression_model = LinearRegression()
        self.__x_train: np.array = None
        self.__x_test: np.array = None
        self.__y_train: np.array = None
        self.__y_test: np.array = None
        self.__predictable_column = ''

        self.__tab_view = tab_view
        self.__tab_view.grid_columnconfigure((0, 1, 2), weight=1)

        # Declaring widgets
        self.__predicted_value_label: customtkinter.CTkLabel
        self.__feature_entries: list[customtkinter.CTkEntry] = []
        self.__feature_entry_labels: list[customtkinter.CTkLabel] = []
        self.__canvas_widget = None

        self.__predicted_value = None
        self.__evaluation_metrics = {}
        self.row_index = 0

    def __invalidate_widgets(self):
        for widgets in self.__tab_view.winfo_children():
            widgets.destroy()

        self.__predicted_value_label = customtkinter.CTkLabel(self.__tab_view)
        self.__feature_entries = []
        self.__feature_entry_labels = []
        self.__canvas_widget = None

        self.__predicted_value = None
        self.__evaluation_metrics = {}
    
    def __create_layout(self):
        # display layout based on the imported dataset
        attribute_list: list[str] = list(self.__dataset.columns.values)
        attribute_list.remove(self.__predictable_column)

        for column in attribute_list:
            self.__feature_entry_labels.append(customtkinter.CTkLabel(self.__tab_view, text=column, font=customtkinter.CTkFont(size=15, weight='bold')))
            self.__feature_entries.append(customtkinter.CTkEntry(self.__tab_view, placeholder_text=f'Enter {column}', width=200))

        self.row_index = 0
        column_index = 0

        for label, entry in zip(self.__feature_entry_labels,  self.__feature_entries):
            label.grid(row=self.row_index, column=column_index, padx=10, pady=(0, 0))
            entry.grid(row=self.row_index + 1, column=column_index, padx=10, pady=(0, 10))
            column_index += 1
            if column_index == 3:
                column_index = 0
                self.row_index += 2

        if column_index != 0:
            self.row_index += 2

        # Training the model
        x = self.__dataset.drop(self.__predictable_column, axis=1).values
        y = self.__dataset[self.__predictable_column].values

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    def __plot(self):
        assert not self.__dataset.empty

        if self.__canvas_widget is not None:
            self.__canvas_widget.destroy()
            print('Destroyed canvas MLR')

        self.__tab_view.grid_rowconfigure(self.row_index + 1, weight=1)

        figure = plt.Figure()
        figure.set_layout_engine("constrained")
        ax = figure.subplots()

        y_predicted = self.__regression_model.predict(self.__x_test)
        comparison_df = pd.DataFrame(data={'Y_Actual': self.__y_test, 'Y_Predicted': y_predicted}, index=range(0, len(y_predicted)))
        y_difference = comparison_df.eval("Y_Predicted - Y_Actual").rename("Y_Difference")

        sns.histplot(y_difference, kde=True, ax=ax)
        ax.yaxis.set_label_position("right")

        canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
        canvas.draw()
        self.__canvas_widget = canvas.get_tk_widget()
        self.__canvas_widget.grid(row=self.row_index + 1, column=0, columnspan=3, sticky='NSEW')

    def get_evaluation_metrics(self) -> dict:
        return self.__evaluation_metrics

    def train_model(self) -> dict:
        self.__regression_model.fit(self.__x_train, self.__y_train)  # Train Model
        self.__plot()  # Plot

        # Evaluate Model
        y_predicted = self.__regression_model.predict(self.__x_test)
        r2score = r2_score(self.__y_test, y_predicted)
        mse = mean_squared_error(self.__y_test, y_predicted)
        rmse = np.sqrt(mse)

        self.__evaluation_metrics["r2_score"] = r2score
        self.__evaluation_metrics["mse"] = mse
        self.__evaluation_metrics["rmse"] = rmse
        return self.__evaluation_metrics

    def predict(self) -> float:
        assert not self.__dataset.empty
        feature_list = [[float(entry.get()) for entry in self.__feature_entries]]
        self.__predicted_value = self.__regression_model.predict(feature_list)[0]
        self.__predicted_value_label.configure(text=f'Predicted {self.__predictable_column}: ₹{round(self.__predicted_value, 2)}',
                                               font=customtkinter.CTkFont(size=20, weight="bold"))
        self.__predicted_value_label.grid(row=self.row_index, column=0,
                                          columnspan=3, padx=10, pady=(0, 10), sticky='WE')
        return self.__predicted_value

    def invalidate(self, dataset: pd.DataFrame, predictable_column: str):
        self.__dataset = dataset
        self.__predictable_column = predictable_column

        self.__invalidate_widgets()
        self.__create_layout()
