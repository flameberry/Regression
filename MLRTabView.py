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

        self.__tab_view = tab_view

        # Declaring widgets
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
        x = self.__dataset.drop('Profit', axis=1).values
        y = self.__dataset['Profit'].values

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=0.3,
                                                                                        random_state=42)
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
            canvas.get_tk_widget().grid(row=3, column=0, columnspan=self.__dataset.columns.size - 1)

    def predict(self):
        if self.dataset_path != '':
            feature_list = [[float(entry.get()) for entry in self.__feature_entries]]
            predicted_value = self.__regression_model.predict(feature_list)
            print(predicted_value)
            self.__predicted_value_label.configure(text=f'Predicted Value: {predicted_value[0]}',
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
            self.__predicted_value_label.grid(row=2, column=0, columnspan=self.__dataset.columns.size - 1, padx=10, pady=(0, 10), sticky='WE')

    def accuracy(self):
        if self.dataset_path != '':
            y_predicted = self.__regression_model.predict(self.__x_test)
            r2score = r2_score(self.__y_test, y_predicted)
            mse = mean_squared_error(self.__y_test, y_predicted)
            rmse = np.sqrt(mse)
            print(f'Multiple Linear Regression: R2_Score: {r2score}, RMSE: {rmse}, MSE: {mse}')

    def invalidate(self, dataset_path: str):
        self.__invalidate_widgets()
        self.__create_layout(dataset_path)
