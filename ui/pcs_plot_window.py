# ui/pcs_plot_window.py

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ui.document_integrations import active_document_title


def open_pcs_plot_popup(state):
    """
    Open or focus a duplicate PCS plot window.
    This popup uses the same figure size and dpi as the main PCS plot.
    """
    root = state["root"]
    win = state.get("pcs_plot_popup")

    if win is not None:
        try:
            if win.winfo_exists():
                win.title(active_document_title(state, "2D PCS"))
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    win = tk.Toplevel(root)
    win.title(active_document_title(state, "2D PCS"))
    win.geometry("620x700")

    frame = tk.Frame(win)
    frame.pack(fill=tk.BOTH, expand=True)

    # Embed directly: pyplot.figure() also creates an independent hidden Tk root
    # whose lifetime outlasts this popup and keeps the mainloop alive on exit.
    fig = Figure(figsize=(4, 4), dpi=150)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()

    state["pcs_plot_popup"] = win
    state["pcs_figure_popup"] = fig
    state["pcs_canvas_popup"] = canvas
    state["pcs_popup_click_cid"] = None
    # Draw the current PCS plot immediately when the popup opens.
    try:
        if "update_graph" in state:
            state["update_graph"]()
    except Exception:
        pass

    def _on_close():
        try:
            win.destroy()
        finally:
            state["pcs_plot_popup"] = None
            state["pcs_figure_popup"] = None
            state["pcs_canvas_popup"] = None
            state["pcs_popup_click_cid"] = None
    win.protocol("WM_DELETE_WINDOW", _on_close)
