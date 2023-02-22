import customtkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime


class RFRTabView:
    """
    This class contains a Random Forest Regressor from sklearn
    to train on multiple columns of a dataset and predict values
    """
    __tab_name = 'RFR'

    @staticmethod
    def get_tab_name() -> str:
        return RFRTabView.__tab_name

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.__dataset: pd.DataFrame
        self.__regression_model = RandomForestRegressor(n_estimators=100, random_state=1)
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

        self.row_index = 0

    def __invalidate_widgets(self):
        for widgets in self.__tab_view.winfo_children():
            widgets.destroy()

        self.__predicted_value_label = customtkinter.CTkLabel(self.__tab_view)
        self.__feature_entries = []
        self.__feature_entry_labels = []

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

        # Define Grid
        grid = {
            'n_estimators': [200, 300, 400, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [3, 4, 5, 6, 7],
            'random_state': [18]
        }

        self.__regression_model.fit(self.__x_train, self.__y_train)
        self.__plot()

    def __plot(self):
        if not self.__dataset.empty:
            self.__tab_view.grid_rowconfigure(self.row_index + 1, weight=1)
            figure = plt.Figure()
            figure.set_layout_engine("constrained")
            ax = figure.subplots()

            # y_predicted = self.__regression_model.predict(self.__x_test)
            # comparison_df = pd.DataFrame(data={'Y_Actual': self.__y_test, 'Y_Predicted': y_predicted}, index=range(0, len(y_predicted)))
            # y_difference = comparison_df.eval("Y_Predicted - Y_Actual").rename("Y_Difference")
            #
            # sns.histplot(y_difference, kde=True, ax=ax)
            # ax.yaxis.set_label_position("right")

            attribute_list: list[str] = list(self.__dataset.columns.values)
            attribute_list.remove(self.__predictable_column)

            model_ranks = pd.Series(self.__regression_model.feature_importances_, index=attribute_list, name='Importance').sort_values(ascending=True, inplace=False)
            model_ranks.plot(ax=ax, kind='barh')

            canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
            canvas.draw()
            canvas.get_tk_widget().grid(row=self.row_index + 1, column=0, columnspan=3, sticky='NSEW')

    def predict(self):
        if not self.__dataset.empty:
            feature_list = [[float(entry.get()) for entry in self.__feature_entries]]
            predicted_value = self.__regression_model.predict(feature_list)
            self.__predicted_value_label.configure(text=f'Predicted {self.__predictable_column}: {predicted_value[0]}',
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
            self.__predicted_value_label.grid(row=self.row_index, column=0,
                                              columnspan=3, padx=10, pady=(0, 10), sticky='WE')

    def accuracy(self):
        if not self.__dataset.empty:
            y_predicted = self.__regression_model.predict(self.__x_test)

            r2score = r2_score(self.__y_test, y_predicted)
            mse = mean_squared_error(self.__y_test, y_predicted)
            rmse = np.sqrt(mse)
            print(f'Random Forest Regression: R2_Score: {r2score * 100}%, RMSE: {rmse}, MSE: {mse}')

    def invalidate(self, dataset: pd.DataFrame, predictable_column: str):
        self.__dataset = dataset
        self.__predictable_column = predictable_column

        self.__invalidate_widgets()
        self.__create_layout()
