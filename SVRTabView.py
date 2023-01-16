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

    def __create_layout(self, dataset_path: str):
        self.dataset_path = dataset_path

        # importing and processing dataset
        self.__dataset = pd.read_csv(self.dataset_path)
        self.__dataset = self.__dataset.select_dtypes(include=np.number)
        self.__dataset.dropna(inplace=True)

        # display layout based on the imported dataset
        for i in range(self.__dataset.columns.size - 1):
            self.__feature_entry_labels.append(customtkinter.CTkLabel(self.__tab_view,
                                                                      text=self.__dataset.columns[i],
                                                                      font=customtkinter.CTkFont(size=15,
                                                                                                 weight='bold')))
            self.__feature_entries.append(customtkinter.CTkEntry(self.__tab_view,
                                                                 placeholder_text=f'Enter {self.__dataset.columns[i]}',
                                                                 width=200))
            self.__feature_entry_labels[i].grid(row=0, column=i, padx=10, pady=(0, 10))
            self.__feature_entries[i].grid(row=1, column=i, padx=10, pady=(0, 10))

        # Training the model
        feature_column_count = 1

        x = self.__dataset.drop('Profit', axis=1).values
        # x = self.__dataset['rdspend'].values
        y = self.__dataset['Profit'].values

        self.__x_standard_scalar = StandardScaler()
        self.__y_standard_scalar = StandardScaler()
        x = self.__x_standard_scalar.fit_transform(x.reshape((len(x), self.__dataset.columns.size - 1)))
        # x = self.__x_standard_scalar.fit_transform(x.reshape((len(x), feature_column_count)))
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
        canvas.get_tk_widget().grid(row=3, column=0, columnspan=self.__dataset.columns.size - 1)

    def predict(self):
        if self.dataset_path != '':
            feature_list = [[float(entry.get()) for entry in self.__feature_entries]]

            predicted_value = self.__regression_model.predict(self.__x_standard_scalar.transform(feature_list))
            predicted_value = self.__y_standard_scalar.inverse_transform([predicted_value])

            self.__predicted_value_label.configure(text=f'Predicted Value: {predicted_value[0][0]}',
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
            self.__predicted_value_label.grid(row=2, column=0, columnspan=self.__dataset.columns.size - 1, padx=10, pady=(0, 10), sticky='WE')

    def accuracy(self):
        if self.dataset_path != '':
            y_predicted = self.__regression_model.predict(self.__x_test)

            y_predicted = self.__y_standard_scalar.inverse_transform(y_predicted.reshape(-1, 1))
            y_test = self.__y_standard_scalar.inverse_transform(self.__y_test.reshape(-1, 1))

            r2score = r2_score(y_test, y_predicted)
            mse = mean_squared_error(y_test, y_predicted)
            rmse = np.sqrt(mse)
            print(f'Support Vector Regression: R2_Score: {r2score}, RMSE: {rmse}, MSE: {mse}')

    def invalidate(self, dataset_path: str):
        self.__invalidate_widgets()
        self.__create_layout(dataset_path)
