"""Presentation of the residual diagnostics; computations remain in logic."""

import tkinter as tk
from tkinter import font as tkfont, ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui.widget_style import ui_font_name


def build_rhombicity_panel(parent, state):
    split = ttk.Panedwindow(parent, orient=tk.VERTICAL, style="PCS.TPanedwindow")
    split.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    upper = ttk.Frame(split, width=1, height=1)
    lower = ttk.Frame(split, width=1, height=1)
    upper.pack_propagate(False)
    lower.pack_propagate(False)
    split.add(upper, weight=65)
    split.add(lower, weight=35)
    state["diag_split"] = split

    figure = Figure(figsize=(4, 3), dpi=100, layout="tight")
    fonts = state.get("fonts", {}) or {}
    figure.text(0.5, 0.60, "Rhombicity Check", ha="center", va="center",
                fontsize=fonts.get("plot_title", 10), fontweight="bold")
    figure.text(
        0.5, 0.43,
        "Load a structure and assign experimental chemical shifts (δ_Exp).\n"
        "Then click Update to run the rhombicity diagnostic.",
        ha="center", va="center", wrap=True, linespacing=1.6,
        fontsize=fonts.get("plot_label", 9), alpha=0.72)
    # The existing plot refresh clears the figure, including this initial guide.
    canvas = FigureCanvasTkAgg(figure, master=upper)
    canvas.get_tk_widget().configure(width=1, height=1)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    state["diag_fig_resphi"] = figure
    state["diag_canvas_resphi"] = canvas

    def fit_axis_labels(_event=None):
        height = canvas.get_tk_widget().winfo_height()
        changed = False
        for ax in figure.axes:
            size = ax.yaxis.label.get_fontsize()
            full = r"Residual, $\delta_{Exp}-\delta_{Pred}$ (ppm)"
            compact = "Residual\n" + r"$\delta_{Exp}-\delta_{Pred}$" + "\n(ppm)"
            label = compact if height < size * figure.dpi / 72 * 18 else full
            if ax.get_ylabel() != label:
                ax.set_ylabel(label, fontsize=size)
                changed = True
        if changed and getattr(_event, "name", None) == "draw_event":
            # A plot refresh can replace the axes without resizing the canvas.
            canvas.get_tk_widget().after_idle(canvas.draw_idle)

    canvas.mpl_connect("resize_event", fit_axis_labels)
    canvas.mpl_connect("draw_event", fit_axis_labels)

    ttk.Label(lower, text="Diagnostic Results", font=ui_font_name(lower, "ui_bold")).pack(
        anchor="w", pady=(2, 4))
    body = ttk.Frame(lower)
    body.pack(fill=tk.BOTH, expand=True)
    box = tk.Text(body, font=ui_font_name(body, "report"), height=6, width=1,
                  padx=10, pady=8, wrap="word", tabstyle="wordprocessor")
    scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=box.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    box.configure(yscrollcommand=scrollbar.set)
    box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    state["diag_result_box"] = box

    def align_columns(_event=None):
        font = tkfont.Font(root=parent._root(), font=box.cget("font"))
        label_width = font.measure("Estimated Δχ_rh") + 20
        # Reserve room for both models; long details below wrap independently.
        half = max(label_width + font.measure("δ = k·G_ax (b = 0)") + 30,
                   (box.winfo_width() - 20) // 2)
        box.configure(tabs=(label_width, half, half + label_width))
        detail_width = font.measure("+1.234567e-03") + 16
        box.tag_configure("details", tabs=tuple(detail_width * i for i in range(1, 7)))
        bold = ui_font_name(box, "ui_bold")
        box.tag_configure("section", font=bold)

    box.bind("<Configure>", align_columns, add="+")
    box.bind("<<ThemeChanged>>", align_columns, add="+")

    geometry = {"height": 0, "ratio": 0.65, "pending": None}

    def constrain_sash():
        geometry["pending"] = None
        height = split.winfo_height()
        if height <= 1:
            return
        font = tkfont.Font(root=parent._root(), font=box.cget("font"))
        minimum_result = max(80, font.metrics("linespace") * 4 + 30)
        minimum_plot = 140
        if height < minimum_plot + minimum_result:
            minimum_plot = round(height * 0.55)
            minimum_result = height - minimum_plot - 8
        y = split.sashpos(0)
        if height != geometry["height"]:
            y = round(height * geometry["ratio"])
        y = max(minimum_plot, min(height - minimum_result - 8, y))
        split.sashpos(0, y)
        geometry.update(height=height, ratio=y / height)

    def schedule_layout(_event=None):
        if geometry["pending"] is None:
            geometry["pending"] = split.after_idle(constrain_sash)

    split.bind("<Configure>", schedule_layout, add="+")
    split.bind("<B1-Motion>", schedule_layout, add="+")
    split.bind("<ButtonRelease-1>", schedule_layout, add="+")
