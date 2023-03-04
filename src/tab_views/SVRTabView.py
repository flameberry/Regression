import customtkinter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVRTabView:
    """
    This class contains a Support Vector Regressor from sklearn
    to train on multiple columns of a dataset and predict values
    """
    __tab_name = 'SVR'

    @staticmethod
    def get_tab_name() -> str:
        return SVRTabView.__tab_name

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.__dataset: pd.DataFrame
        self.__regression_model = SVR(kernel='rbf')
        self.__x_train: np.array = None
        self.__x_test: np.array = None
        self.__y_train: np.array = None
        self.__y_test: np.array = None
        self.__x_standard_scalar: StandardScaler
        self.__y_standard_scalar: StandardScaler

        self.__predictable_column = ''
        self.row_index = 0

        self.__tab_view = tab_view
        self.__tab_view.grid_columnconfigure((0, 1, 2), weight=1)

        # defining widgets
        self.__predicted_value_label: customtkinter.CTkLabel
        self.__feature_entries: list[customtkinter.CTkEntry] = []
        self.__feature_entry_labels: list[customtkinter.CTkLabel] = []
        self.__canvas_widget = None

        self.__predicted_value = None
        self.__evaluation_metrics = {}

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
        attribute_list: list[str] = list(self.__dataset.columns.values)
        attribute_list.remove(self.__predictable_column)

        for column in attribute_list:
            self.__feature_entry_labels.append(customtkinter.CTkLabel(self.__tab_view, text=column, font=customtkinter.CTkFont(size=15, weight='bold')))
            self.__feature_entries.append(customtkinter.CTkEntry(self.__tab_view, placeholder_text=f'Enter {column}', width=200))

        self.row_index = 0
        column_index = 0

        for label, entry in zip(self.__feature_entry_labels, self.__feature_entries):
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

        self.__x_standard_scalar = StandardScaler()
        self.__y_standard_scalar = StandardScaler()
        x = self.__x_standard_scalar.fit_transform(x.reshape((len(x), self.__dataset.columns.size - 1)))
        y = self.__y_standard_scalar.fit_transform(y.reshape((len(y), 1)))

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=0.3,
                                                                                        random_state=42)

    def __plot(self):
        assert not self.__dataset.empty
        if self.__canvas_widget is not None:
            self.__canvas_widget.destroy()
            print('Destroyed canvas SVR')

        self.__tab_view.grid_rowconfigure(self.row_index + 1, weight=1)

        figure = plt.Figure()
        figure.set_layout_engine("constrained")
        ax = figure.subplots()

        y_predicted = self.__regression_model.predict(self.__x_test)
        y_predicted = self.__y_standard_scalar.inverse_transform(y_predicted.reshape(-1, 1))
        y_test = self.__y_standard_scalar.inverse_transform(self.__y_test.reshape(-1, 1))

        comparison_df = pd.DataFrame(data={'Y_Actual': y_test.ravel(), 'Y_Predicted': y_predicted.ravel()})
        y_difference = comparison_df.eval("Y_Predicted - Y_Actual").rename("Y_Difference")

        sns.histplot(y_difference, kde=True, ax=ax)
        ax.yaxis.set_label_position("right")

        canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
        canvas.draw()
        self.__canvas_widget = canvas.get_tk_widget()
        self.__canvas_widget.grid(row=self.row_index + 1, column=0, columnspan=3, sticky='NSEW')

    def train_model(self) -> dict:
        self.__regression_model.fit(self.__x_train, self.__y_train.ravel())
        self.__plot()

        # Evaluate Model
        y_predicted = self.__regression_model.predict(self.__x_test)
        y_predicted = self.__y_standard_scalar.inverse_transform(y_predicted.reshape(-1, 1))
        y_test = self.__y_standard_scalar.inverse_transform(self.__y_test.reshape(-1, 1))

        r2score = r2_score(y_test, y_predicted)
        mse = mean_squared_error(y_test, y_predicted)
        rmse = np.sqrt(mse)

        self.__evaluation_metrics["r2_score"] = r2score
        self.__evaluation_metrics["mse"] = mse
        self.__evaluation_metrics["rmse"] = rmse
        return self.__evaluation_metrics

    def predict(self) -> float:
        assert not self.__dataset.empty
        feature_list = [[float(entry.get()) for entry in self.__feature_entries]]

        self.__predicted_value = self.__regression_model.predict(self.__x_standard_scalar.transform(feature_list))
        self.__predicted_value = self.__y_standard_scalar.inverse_transform([self.__predicted_value])
        self.__predicted_value = self.__predicted_value[0][0]

        self.__predicted_value_label.configure(text=f'Predicted Value: {self.__predicted_value}', font=customtkinter.CTkFont(size=20, weight="bold"))
        self.__predicted_value_label.grid(row=self.row_index, column=0, columnspan=3, padx=10, pady=(0, 10), sticky='WE')
        return self.__predicted_value

    def get_evaluation_metrics(self) -> dict:
        return self.__evaluation_metrics

    def invalidate(self, dataset: pd.DataFrame, predictable_column: str):
        self.__dataset = dataset
        self.__predictable_column = predictable_column

        self.__invalidate_widgets()
        self.__create_layout()
