import platform
import pathlib
import tkinter
import customtkinter
from PIL import Image
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tab_views.DatasetTabView import DatasetTabView
from tab_views.LRTabView import LRTabView
from tab_views.MLRTabView import MLRTabView
from tab_views.SVRTabView import SVRTabView
from tab_views.RFRTabView import RFRTabView
from tab_views.NNRTabView import NNRTabView

from utils import center, project_dir


class RegressionApp(customtkinter.CTk):
    """
    This class contains the complete user interface of the app
    to compare multiple regression methods
    """
    def __init__(self, **kwargs):
        """
        creates the main app layout for regression
        param - methods: a tuple of `Regression` enum values,
        to specify the methods to be shown in the user interface
        """
        super().__init__()

        self.__dataset_path: str = ''
        self.__dataset: pd.DataFrame
        self.__predictable_col_string_var = None

        self.__loading_widget = None
        self.__loading_widget_label: customtkinter.CTkLabel
        self.__progress_bar: customtkinter.CTkProgressBar

        self.__window_width = min(900, self.winfo_screenwidth())
        self.__window_height = min(700, self.winfo_screenheight())

        self.__title = 'Regression'

        self.__frame_corner_radius = 5
        self.__frame_padding = 10

        self.title(self.__title)
        self.geometry(f"{self.__window_width}x{self.__window_height}")

        center(self)

        self.__menu_bar: tkinter.Menu

        if platform.system() == 'Darwin':
            self.__create_menu_bar()
            
        self.__setup_shortcuts()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self.__create_toolbar_layout()

        self.__sidebar_frame = customtkinter.CTkFrame(self, width=100, corner_radius=self.__frame_corner_radius)
        self.__sidebar_frame.grid(row=0, column=0, rowspan=3, padx=(self.__frame_padding, 0), pady=self.__frame_padding, sticky="nsew")
        self.__sidebar_frame.grid_rowconfigure(5, weight=1)

        self.__project_name_label = customtkinter.CTkLabel(self.__sidebar_frame, text='Flameberry', font=customtkinter.CTkFont(size=17, weight='bold'))
        self.__project_name_label.grid(row=0, column=0)

        self.__dataset_name_label = customtkinter.CTkLabel(self.__sidebar_frame, text='No Dataset Loaded', wraplength=150,
                                                           font=customtkinter.CTkFont(weight='bold'))
        self.__dataset_name_label.grid(row=7, column=0, padx=20, pady=10)

        self.__main_tab_view = customtkinter.CTkTabview(self, corner_radius=self.__frame_corner_radius, command=self.__update_layout_on_tab_change)
        self.__main_tab_view.grid(row=1, column=1, padx=self.__frame_padding, pady=(0, self.__frame_padding), sticky="NSEW")

        if 'methods' in kwargs:
            assert type(kwargs['methods']) == tuple, "`methods` parameter should be a tuple!"
            assert len(kwargs['methods']) != 0, "`methods` should not be an empty tuple!"
            self.__tab_view_types = [DatasetTabView] + [method.tab_type() for method in kwargs['methods']]
        else:
            # self.__tab_view_types = [DatasetTabView, LRTabView, MLRTabView, SVRTabView, RFRTabView, NNRTabView]
            self.__tab_view_types = [DatasetTabView, LRTabView, MLRTabView, SVRTabView, RFRTabView]

        self.__tab_views = []
        for tab_view_type in self.__tab_view_types:
            tab_name = tab_view_type.get_tab_name()
            self.__main_tab_view.add(tab_name)
            self.__tab_views.append(tab_view_type(self.__main_tab_view.tab(tab_name)))
        
        # Create Status Bar
        self.__status_bar = customtkinter.CTkFrame(self, height=30, corner_radius=self.__frame_corner_radius)
        self.__status_bar.grid(row=2, column=1, padx=self.__frame_padding, pady=(0, self.__frame_padding), sticky='NSEW')

        self.__evaluation_metric_label = customtkinter.CTkLabel(self.__status_bar, text='')
        self.__evaluation_metric_label.grid(row=0, column=0, padx=10)

        plt.style.use("seaborn-dark")

    def __import_dataset(self, *args):
        path = filedialog.askopenfilename(
            title='Select dataset file',
            initialdir=pathlib.Path(__file__).parent.parent,
            filetypes=(("CSV", "*.csv"), ("XLS Files", "*.xls"), ("XLSX Files", "*.xlsx"), ("All Files", "*.*"))
        )
        if path == '':
            return

        if self.__dataset_path != path:
            self.__dataset_path = path
            file_name = path.split('/')[-1]
            self.__dataset_name_label.configure(text=file_name)

            self.__reload_dataset()

            # Update window title to show currently loaded dataset
            self.__title = f'Regression - {file_name}'
            self.title(self.__title)

    def __reload_dataset(self, *args):
        if len(self.__dataset_path):
            self.__create_loading_widget()
            extension = self.__dataset_path.split('.')[-1]
            if extension == 'csv':
                self.__dataset = pd.read_csv(self.__dataset_path)
            # elif extension == 'xlsx' or extension == 'xls':  # TODO: Fixme
            #     self.__dataset = pd.read_excel(self.__dataset_path)
            self.__dataset = self.__dataset.select_dtypes(include=np.number)
            self.__dataset.dropna(inplace=True)

            attribute_list = list(self.__dataset.columns.values)

            self.__predictable_col_string_var = tkinter.StringVar()
            self.__predictable_col_string_var.set(attribute_list[-1])

            # Toolbar
            self.__predictable_column_option_menu_toolbar.configure(values=attribute_list,
                                                                    variable=self.__predictable_col_string_var)

            self.__predictable_col_string_var.trace("w", self.__predictable_column_callback)
            selected_col = self.__predictable_col_string_var.get()

            self.__evaluation_metric_label.configure(text='')

            tab_view_count = len(self.__tab_views)
            finished_count = 0
            for tab_view in self.__tab_views:
                self.__loading_widget_label.configure(text=f'Creating {type(tab_view).__name__}....')
                self.__progress_bar.set(value=finished_count / tab_view_count)
                self.__loading_widget.update_idletasks()
                tab_view.invalidate(self.__dataset, selected_col)
                finished_count += 1

            self.__loading_widget_label.configure(text='Done.')
            self.__progress_bar.set(value=1)
            self.__loading_widget.update_idletasks()

            self.__loading_widget.destroy()
        else:
            print('ERROR: Failed to reload dataset!')

    def __create_loading_widget(self):
        self.__loading_widget = customtkinter.CTkToplevel(self)
        self.__loading_widget.title('Loading Dataset')
        self.__loading_widget.geometry('400x75')
        center(self.__loading_widget, parent=self)

        self.__loading_widget.grab_set()
        self.__loading_widget.transient(self)

        self.__loading_widget_label = customtkinter.CTkLabel(self.__loading_widget, text='Loading dataset...', font=customtkinter.CTkFont(size=15, weight="bold"))
        self.__loading_widget_label.pack(pady=10)

        self.__progress_bar = customtkinter.CTkProgressBar(self.__loading_widget, mode='determinate', width=300)
        self.__progress_bar.pack()

    def __predictable_column_callback(self, *args):
        column = self.__predictable_col_string_var.get()
        for tab_view in self.__tab_views:
            tab_view.invalidate(self.__dataset, column)

    def __update_layout_on_tab_change(self, *args):
        current_tab = self.__main_tab_view.get()
        for tab_view in self.__tab_views:
            if current_tab == type(tab_view).get_tab_name():
                try:
                    evaluation_metrics = tab_view.get_evaluation_metrics()
                except AttributeError:
                    evaluation_metrics = {}
                if len(evaluation_metrics) == 0:
                    self.__evaluation_metric_label.configure(text='')
                else:
                    self.__evaluation_metric_label.configure(
                        text=f'R2_score: {round(evaluation_metrics["r2_score"], 6)}, MSE: {round(evaluation_metrics["mse"], 4)}, RMSE: {round(evaluation_metrics["rmse"], 4)}'
                    )

    def __predict(self, *args):
        current_tab = self.__main_tab_view.get()
        for tab_view in self.__tab_views:
            if current_tab == type(tab_view).get_tab_name():
                predicted_val = tab_view.predict()

    def __train_model(self, *args):
        current_tab = self.__main_tab_view.get()
        for tab_view in self.__tab_views:
            if current_tab == type(tab_view).get_tab_name():
                evaluation_metrics = tab_view.train_model()
                self.__evaluation_metric_label.configure(text=f'R2_score: {round(evaluation_metrics["r2_score"], 6)}, MSE: {round(evaluation_metrics["mse"], 4)}, RMSE: {round(evaluation_metrics["rmse"], 4)}')
            else:
                try:
                    tab_view.train_model()
                except AttributeError:
                    pass
    
    def __save(self, *args):
        current_tab = self.__main_tab_view.get()
        if current_tab == NNRTabView.get_tab_name():
            for tab_view in self.__tab_views:
                if current_tab == type(tab_view).get_tab_name():
                    file_name = str(self.__dataset_path.split('/')[-1]).split('.')[0]
                    tab_view.save(file_name)

    def __setup_shortcuts(self):
        super_key = 'Command' if platform.system() == 'Darwin' else 'Control'

        self.bind_all(f"<{super_key}-o>", func=self.__import_dataset)
        self.bind_all(f"<{super_key}-y>", func=self.__predict)
        self.bind_all(f"<{super_key}-r>", func=self.__reload_dataset)
        self.bind_all(f"<{super_key}-t>", func=self.__train_model)
        self.bind_all(f"<{super_key}-s>", func=self.__save)

    def __create_menu_bar(self):
        """
        Only for macOS
        """
        self.__menu_bar = tkinter.Menu(self)

        self.createcommand("tk::mac::Quit", self.destroy)
        self.createcommand("tk::mac::ShowPreferences", lambda: print("Preferences"))

        file_menu = tkinter.Menu(self.__menu_bar, tearoff=0)
        self.__menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tkinter.Menu(self.__menu_bar, tearoff=0)
        self.__menu_bar.add_cascade(label="Edit", menu=edit_menu)

        file_menu.add_command(label="Load Dataset", command=self.__import_dataset, accelerator="Command-O")

        edit_menu.add_command(label="Undo")
        edit_menu.add_command(label="Redo")

        tools_menu = tkinter.Menu(self.__menu_bar, tearoff=0)
        self.__menu_bar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Predict", accelerator="Command-Y", command=self.__predict)
        tools_menu.add_command(label="Train All Models", accelerator="Command-T", command=self.__train_model)
        tools_menu.add_command(label="Reload Dataset", accelerator="Command-R", command=self.__reload_dataset)

        self.configure(menu=self.__menu_bar)

    def __create_toolbar_layout(self):
        self.__tool_bar_frame = customtkinter.CTkFrame(self, corner_radius=self.__frame_corner_radius)
        self.__tool_bar_frame.grid(row=0, column=1, padx=self.__frame_padding, pady=(self.__frame_padding, 0), sticky='NSEW')

        self.__tool_bar_frame.grid_rowconfigure(0, weight=1)
        self.__tool_bar_frame.grid_columnconfigure(2, weight=1)

        import_button_image = customtkinter.CTkImage(
            Image.open(str(project_dir) + '/icons/add_file_icon_24x24.png'),
            size=(24, 24)
        )

        self.__import_button_tool_bar = customtkinter.CTkButton(
            self.__tool_bar_frame,
            image=import_button_image,
            text='',
            command=self.__import_dataset,
            width=24,
            height=24,
            corner_radius=5
        )
        self.__import_button_tool_bar.grid(row=0, column=0, sticky='NS')

        reload_button_icon_image = customtkinter.CTkImage(
            Image.open(str(project_dir) + '/icons/reload_icon_24x24.png'),
            size=(24, 24)
        )

        self.__reload_button_toolbar = customtkinter.CTkButton(
            self.__tool_bar_frame,
            image=reload_button_icon_image,
            text='',
            command=self.__reload_dataset,
            width=24,
            height=24,
            corner_radius=5
        )
        self.__reload_button_toolbar.grid(row=0, column=1, padx=(5, 0), sticky='NS')

        self.__predictable_column_option_menu_toolbar = customtkinter.CTkOptionMenu(self.__tool_bar_frame, values=[])
        self.__predictable_column_option_menu_toolbar.set('N/A')
        self.__predictable_column_option_menu_toolbar.grid(row=0, column=2)

        hammer_icon_image = customtkinter.CTkImage(
            Image.open(str(project_dir) + '/icons/hammer_icon_48x48.png'),
            size=(24, 24)
        )

        self.__train_button_toolbar = customtkinter.CTkButton(
            self.__tool_bar_frame,
            image=hammer_icon_image,
            text='',
            command=self.__train_model,
            width=24,
            height=24,
            corner_radius=5
        )
        self.__train_button_toolbar.grid(row=0, column=3, sticky='NS')

        play_button_image = customtkinter.CTkImage(
            Image.open(str(project_dir) + '/icons/play_button_icon.png'),
            size=(24, 24)
        )

        self.__play_button = customtkinter.CTkButton(
            self.__tool_bar_frame,
            image=play_button_image,
            text='',
            command=self.__predict,
            width=24,
            height=24,
            corner_radius=5
        )
        self.__play_button.grid(row=0, column=4, padx=(5, 0), sticky='NS')
