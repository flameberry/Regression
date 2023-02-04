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
    __tab_name = 'Support Vector Regression'

    @staticmethod
    def get_tab_name() -> str:
        return SVRTabView.__tab_name

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.dataset_path = ''

        self.__dataset = pd.DataFrame()
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

        # defining widgets
        self.__predicted_value_label: customtkinter.CTkLabel
        self.__feature_entries: list[customtkinter.CTkEntry] = []
        self.__feature_entry_labels: list[customtkinter.CTkLabel] = []

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

        attribute_list: list[str] = list(self.__dataset.columns.values)
        attribute_list.remove(self.__predictable_column)

        for column in attribute_list:
            self.__feature_entry_labels.append(customtkinter.CTkLabel(self.__tab_view, text=column, font=customtkinter.CTkFont(size=15, weight='bold')))
            self.__feature_entries.append(customtkinter.CTkEntry(self.__tab_view, placeholder_text=f'Enter {column}', width=200))

        self.row_index = 0
        column_index = 0

        for label, entry in zip(self.__feature_entry_labels, self.__feature_entries):
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

        self.__x_standard_scalar = StandardScaler()
        self.__y_standard_scalar = StandardScaler()
        x = self.__x_standard_scalar.fit_transform(x.reshape((len(x), self.__dataset.columns.size - 1)))
        y = self.__y_standard_scalar.fit_transform(y.reshape((len(y), 1)))

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=0.3,
                                                                                        random_state=42)
        self.__regression_model.fit(self.__x_train, self.__y_train.ravel())
        self.__plot()

    def __plot(self):
        figure = plt.Figure(figsize=(6, 5))
        figure.set_layout_engine("constrained")
        ax = figure.subplots()

        y_predicted = self.__regression_model.predict(self.__x_test)
        y_predicted = self.__y_standard_scalar.inverse_transform(y_predicted.reshape(-1, 1))
        y_test = self.__y_standard_scalar.inverse_transform(self.__y_test.reshape(-1, 1))

        comparison_df = pd.DataFrame(data={'Y_Actual': y_test.ravel(), 'Y_Predicted': y_predicted.ravel()})
        y_difference = comparison_df.eval("Y_Predicted - Y_Actual").rename("Y_Difference")
        diff_range = min(abs(y_difference.min()), abs(y_difference.max()))

        sns.scatterplot(data=comparison_df, x='Y_Actual', y='Y_Predicted', ax=ax, hue=y_difference, size=y_difference,
                        size_norm=(-diff_range, diff_range))
        ax.yaxis.set_label_position("right")

        canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
        canvas.draw()
        canvas.get_tk_widget().grid(row=self.row_index + 1, column=0, columnspan=3)

    def on_change_predictable_column(self, column: str):
        self.__predictable_column = column
        self.invalidate(self.dataset_path, self.__predictable_column)
        print(f'SVR: Set predictable column as: {column}')

    def predict(self):
        if self.dataset_path != '':
            feature_list = [[float(entry.get()) for entry in self.__feature_entries]]

            predicted_value = self.__regression_model.predict(self.__x_standard_scalar.transform(feature_list))
            predicted_value = self.__y_standard_scalar.inverse_transform([predicted_value])

            self.__predicted_value_label.configure(text=f'Predicted Value: {predicted_value[0][0]}', font=customtkinter.CTkFont(size=20, weight="bold"))
            self.__predicted_value_label.grid(row=self.row_index, column=0, columnspan=3, padx=10, pady=(0, 10), sticky='WE')

    def accuracy(self):
        if self.dataset_path != '':
            y_predicted = self.__regression_model.predict(self.__x_test)

            y_predicted = self.__y_standard_scalar.inverse_transform(y_predicted.reshape(-1, 1))
            y_test = self.__y_standard_scalar.inverse_transform(self.__y_test.reshape(-1, 1))

            r2score = r2_score(y_test, y_predicted)
            mse = mean_squared_error(y_test, y_predicted)
            rmse = np.sqrt(mse)
            print(f'Support Vector Regression: R2_Score: {r2score * 100}%, RMSE: {rmse}, MSE: {mse}')

    def invalidate(self, dataset_path: str, predictable_column: str):
        reload = self.dataset_path != dataset_path

        self.dataset_path = dataset_path
        self.__predictable_column = predictable_column

        self.__invalidate_widgets()
        self.__create_layout(reload_dataset=reload)
