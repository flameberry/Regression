import datetime

import customtkinter
from tkinter import filedialog
import pathlib
from utils import project_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

import tensorflow as tf
import keras
from keras import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# x_train = features, y_train = target
# train_data = pd.read_csv(TRAIN_DATA_PATH)
# test_data = pd.read_csv(TEST_DATA_PATH)
# x_train, y_train = train_data.drop(TARGET_NAME, axis=1), train_data[TARGET_NAME]
# x_test, y_test = test_data.drop(TARGET_NAME, axis=1), test_data[TARGET_NAME]
#
# """
#   Standard Scale test and train data
#   Z - Score normalization
# """
# def scale_datasets(x_train, x_test):
#     standard_scaler = StandardScaler()
#     x_train_scaled_local = pd.DataFrame(
#         standard_scaler.fit_transform(x_train),
#         columns=x_train.columns
#     )
#     x_test_scaled_local = pd.DataFrame(
#         standard_scaler.transform(x_test),
#         columns=x_test.columns
#     )
#     return x_train_scaled_local, x_test_scaled_local
#
#
# x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
#
# hidden_units1 = 160
# hidden_units2 = 480
# hidden_units3 = 256
# learning_rate = 0.01
#
#
# # Creating model using the Sequential in tensorflow
# def build_model_using_sequential():
#     model = Sequential([
#         Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
#         Dropout(0.2),
#         Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
#         Dropout(0.2),
#         Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
#         Dense(1, kernel_initializer='normal', activation='linear')
#     ])
#     return model
#
#
# # build the model
# model = build_model_using_sequential()
#
# # loss function
# msle = MeanSquaredLogarithmicError()
# model.compile(
#     loss=msle,
#     optimizer=Adam(learning_rate=learning_rate),
#     metrics=[msle]
# )
# # train the model
# history = model.fit(
#     x_train_scaled.values,
#     y_train.values,
#     epochs=10,
#     batch_size=64,
#     validation_split=0.2
# )
#
#
# def plot_history(history, key):
#     plt.plot(history.history[key])
#     plt.plot(history.history['val_' + key])
#     plt.xlabel("Epochs")
#     plt.ylabel(key)
#     plt.legend([key, 'val_' + key])
#     plt.show()
#
#
# # Plot the history
# plot_history(history, 'mean_squared_logarithmic_error')
#
# x_test['prediction'] = model.predict(x_test_scaled)


class NNRTabView:
    """
    This class contains a Deep Neural Network Regressor from tensorflow
    to train on a dataset and predict values
    """
    __tab_name = 'NNR'

    @staticmethod
    def get_tab_name() -> str:
        return NNRTabView.__tab_name

    def __init__(self, tab_view: customtkinter.CTkFrame):
        self.__dataset: pd.DataFrame
        self.__regression_model: Sequential
        self.__x_train: np.array = None
        self.__x_test: np.array = None
        self.__x_train_scaled: np.array = None
        self.__x_test_scaled: np.array = None
        self.__y_train: np.array = None
        self.__y_test: np.array = None
        self.__standard_scaler = StandardScaler()
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
        attribute_list = list(self.__dataset.columns.values)
        attribute_list.remove(self.__predictable_column)

        for column in attribute_list:
            self.__feature_entry_labels.append(customtkinter.CTkLabel(self.__tab_view, text=column,
                                                                      font=customtkinter.CTkFont(size=15,
                                                                                                 weight='bold')))
            self.__feature_entries.append(
                customtkinter.CTkEntry(self.__tab_view, placeholder_text=f'Enter {column}', width=200))

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
        
        self.__create_neural_network()

        # Saving the model for later use
        save_path = str(project_dir) + f'/models/DNNR_Model_{datetime.datetime.now()}'
        self.__regression_model.save(save_path)

        self.__plot()
    
    def __create_neural_network(self):
        # Training the neural network
        x = self.__dataset.drop(self.__predictable_column, axis=1).values
        y = self.__dataset[self.__predictable_column].values

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        self.__x_train_scaled = pd.DataFrame(
            self.__standard_scaler.fit_transform(self.__x_train)
        )
        self.__x_test_scaled = pd.DataFrame(
            self.__standard_scaler.transform(self.__x_test)
        )

        N_TRAIN = len(self.__x_train)
        BATCH_SIZE = 500
        STEPS_PER_EPOCH = max(1, N_TRAIN // BATCH_SIZE)
        MAX_EPOCHS = 1000

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH * 1000,
            decay_rate=1,
            staircase=False
        )

        def get_optimizer():
            return tf.keras.optimizers.Adam(lr_schedule)

        self.__regression_model = Sequential()
        # Input Layer
        self.__regression_model.add(Dense(64, kernel_initializer='normal', input_dim=len(self.__dataset.columns) - 1, activation='relu'))

        # Hidden Layers
        self.__regression_model.add(Dropout(0.2))
        self.__regression_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        self.__regression_model.add(Dropout(0.2))
        # self.__regression_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        # self.__regression_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

        # Output Layer
        self.__regression_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        msle = tf.keras.losses.MeanSquaredLogarithmicError()
        self.__regression_model.compile(
            # loss='mean_squared_error',
            loss=msle,
            optimizer=get_optimizer(),
            metrics=[msle]
            # metrics=['mean_squared_error']
        )

        print(self.__regression_model.summary())

        self.__history = self.__regression_model.fit(
            self.__x_train,
            self.__y_train,
            verbose=1,
            epochs=MAX_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=(self.__x_test, self.__y_test)
        )

    def __plot(self):
        if not self.__dataset.empty:
            self.__tab_view.grid_rowconfigure(self.row_index + 1, weight=1)
            figure = plt.Figure()
            figure.set_layout_engine("constrained")
            ax = figure.subplots()

            # plot the training and validation accuracy and loss at each epoch
            loss = self.__history.history['loss']
            val_loss = self.__history.history['val_loss']
            epochs = range(1, len(loss) + 1)

            sns.lineplot(x=epochs, y=loss, label='Training loss', ax=ax)
            sns.lineplot(x=epochs, y=val_loss, label='Validation loss', ax=ax)
            ax.xaxis.set_label('Epochs')
            ax.yaxis.set_label('Loss')

            # plt.plot(epochs, loss, 'y', label='Training loss')
            # plt.plot(epochs, val_loss, 'r', label='Validation loss')
            # plt.title('Training and validation loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            ax.yaxis.set_label_position("right")

            canvas = FigureCanvasTkAgg(figure, master=self.__tab_view)
            canvas.draw()
            canvas.get_tk_widget().grid(row=self.row_index + 1, column=0, columnspan=3, sticky='NSEW')
    
    def __load_model(self, path: str = None, open_dialog=True):
        if open_dialog:
            path = filedialog.askopenfilename(
                title='Select dataset file',
                initialdir=pathlib.Path(__file__).parent,
                filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
            )
        if path == '':
            return
        
        self.__regression_model = keras.models.load_model(path)
        pass

    def predict(self):
        if not self.__dataset.empty:
            feature_list = [[float(entry.get()) for entry in self.__feature_entries]]
            # self.__standard_scaler.transform(feature_list)
            predicted_value = self.__regression_model.predict(feature_list)
            self.__predicted_value_label.configure(text=f'Predicted {self.__predictable_column}: {predicted_value[0]}',
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
            self.__predicted_value_label.grid(row=self.row_index, column=0,
                                              columnspan=3, padx=10, pady=(0, 10), sticky='WE')

    def accuracy(self):
        prediction_test = self.__regression_model.predict(self.__x_test)
        # print(self.__y_test, prediction_test)
        r2score = r2_score(self.__y_test, prediction_test)
        mse = np.mean(prediction_test - self.__y_test) ** 2
        print(f"Neural Network Regression: R2 Score: {r2score * 100}%, RMSE: {np.sqrt(mse)}, MSE: {mse}")

    def invalidate(self, dataset: pd.DataFrame, predictable_column: str):
        self.__dataset = dataset
        self.__predictable_column = predictable_column

        self.__invalidate_widgets()
        self.__create_layout()
