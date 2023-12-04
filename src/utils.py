import pathlib

project_dir = pathlib.Path(__file__).parent.parent.absolute()


def center(win, parent=None):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    :param parent: the parent window relative to which `win` will be centered
    """
    win.update_idletasks()
    win.withdraw()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()

    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width

    if parent is None:
        x = win.winfo_screenwidth() // 2 - win_width // 2
        y = win.winfo_screenheight() // 2 - win_height // 2
    else:
        x = parent.winfo_width() // 2 - win_width // 2 + parent.winfo_x()
        y = parent.winfo_height() // 2 - win_height // 2 + parent.winfo_y()

    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()
