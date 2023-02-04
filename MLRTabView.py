import customtkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class MLRTabView:
    __tab_name = 'Multiple Linear Regression'

    @staticmethod
    def get_tab_name() -> str:
        return MLRTabView.__tab_name

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.dataset_path = ''

        self.__dataset = pd.DataFrame()
        self.__regression_model = LinearRegression()
        self.__x_train: np.array = None
        self.__x_test: np.array = None
        self.__y_train: np.array = None
        self.__y_test: np.array = None
        self.__predictable_column = ''

        self.__tab_view = tab_view

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

    def __create_layout(self, reload_dataset=True):
        if reload_dataset:
            # importing and processing dataset
            self.__dataset = pd.read_csv(self.dataset_path)
            self.__dataset = self.__dataset.select_dtypes(include=np.number)
            self.__dataset.dropna(inplace=True)

        # display layout based on the imported dataset
        attribute_list: list[str] = list(self.__dataset.columns.values)
        attribute_list.remove(self.__predictable_column)

        for column in attribute_list:
            self.__feature_entry_labels.append(customtkinter.CTkLabel(self.__tab_view, text=column, font=customtkinter.CTkFont(size=15, weight='bold')))
            self.__feature_entries.append(customtkinter.CTkEntry(self.__tab_view, placeholder_text=f'Enter {column}', width=200))

        self.row_index = 0
        column_index = 0

        for label, entry in zip(self.__feature_entry_labels,  self.__feature_entries):
            label.grid(row=self.row_index, column=column_index, padx=10, pady=(0, 10))
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
        self.__regression_model.fit(self.__x_train, self.__y_train)
        self.__plot()

    def __plot(self):
        if self.dataset_path != '':
            figure = plt.Figure(figsize=(6, 5))
            figure.set_layout_engine("constrained")
            ax = figure.subplots()

            y_predicted = self.__regression_model.predict(self.__x_test)
            comparison_df = pd.DataFrame(data={'Y_Actual': self.__y_test, 'Y_Predicted': y_predicted})
            y_difference = comparison_df.eval("Y_Predicted - Y_Actual").rename("Y_Difference")
            diff_range = min(abs(y_difference.min()), abs(y_difference.max()))

            sns.scatterplot(data=comparison_df, x='Y_Actual', y='Y_Predicted', ax=ax, hue=y_difference,
                            size=y_difference, size_norm=(-diff_range, diff_range))
            ax.yaxis.set_label_position("right")

            canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
            canvas.draw()
            canvas.get_tk_widget().grid(row=self.row_index + 1, column=0, columnspan=3)

    def on_change_predictable_column(self, column: str):
        self.__predictable_column = column
        self.invalidate(self.dataset_path, self.__predictable_column)
        print(f'MLR: Set predictable column as: {column}')

    def predict(self):
        if self.dataset_path != '':
            feature_list = [[float(entry.get()) for entry in self.__feature_entries]]
            predicted_value = self.__regression_model.predict(feature_list)
            print(predicted_value)
            self.__predicted_value_label.configure(text=f'Predicted {self.__predictable_column}: {predicted_value[0]}',
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
            self.__predicted_value_label.grid(row=self.row_index, column=0,
                                              columnspan=3, padx=10, pady=(0, 10), sticky='WE')

    def accuracy(self):
        if self.dataset_path != '':
            y_predicted = self.__regression_model.predict(self.__x_test)
            r2score = r2_score(self.__y_test, y_predicted)
            mse = mean_squared_error(self.__y_test, y_predicted)
            rmse = np.sqrt(mse)
            print(f'Multiple Linear Regression: R2_Score: {r2score * 100}%, RMSE: {rmse}, MSE: {mse}')

    def invalidate(self, dataset_path: str, predictable_column: str):
        reload = True
        if self.dataset_path == dataset_path:
            reload = False

        self.dataset_path = dataset_path
        self.__predictable_column = predictable_column

        self.__invalidate_widgets()
        self.__create_layout(reload_dataset=reload)
