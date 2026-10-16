"""Standard fitting layout and saved-condition presentation only."""

import textwrap
import tkinter as tk
from tkinter import ttk

from ui.scrollable_workspace import VerticalScrolledWorkspace
from ui.widget_style import ui_native_colors, ui_font_name, ui_text_color


def build_fitting_workspace(parent, state):
    workspace = VerticalScrolledWorkspace(parent)
    workspace.pack(fill=tk.BOTH, expand=True)
    ui_native_colors(workspace.canvas, background="surface")
    state["fit_workspace_scroll"] = workspace
    parent = workspace.content
    header = tk.Frame(parent, borderwidth=0, highlightthickness=1)
    ui_native_colors(header, background="selected", highlightbackground="accent",
                     highlightcolor="accent")
    header.pack(fill=tk.X, padx=6, pady=(6, 2))
    # Explicit request size prevents pane insertion/removal from propagating
    # transient content dimensions through the complete main-window layout.
    split = ttk.Panedwindow(parent, orient=tk.VERTICAL, style="PCS.TPanedwindow",
                            width=1, height=1)
    split.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
    settings = ttk.Frame(split, width=1, height=1)
    settings.pack_propagate(False)
    output = ttk.Frame(split)
    split.add(settings, weight=0)
    split.add(output, weight=1)
    state["fit_vertical_split"] = split
    geometry = {"height": None, "preferred_height": None, "dragged": False,
                "pending": None, "collapsed": False,
                "saved_height": None}

    def constrain():
        geometry["pending"] = None
        if geometry["collapsed"] or split.winfo_height() <= 1:
            return
        height = split.winfo_height()
        maximum = max(60, height - output.winfo_reqheight() - 8)
        if geometry["preferred_height"] is None:
            geometry["preferred_height"] = state["fit_config_frame"].winfo_reqheight() + 8
        if geometry["dragged"]:
            geometry["preferred_height"] = split.sashpos(0)
            geometry["dragged"] = False
        requested = geometry["preferred_height"]
        minimum = min(state["fit_config_frame"].winfo_reqheight() + 8, maximum)
        position = max(minimum, min(maximum, requested))
        if split.sashpos(0) != position:
            split.sashpos(0, position)
        geometry["height"] = height
        schedule_workspace()

    def fill_workspace():
        workspace_pending["id"] = None
        # Only the complete tab overflows in very short windows. Keep the
        # settings lists usable rather than shrinking them to one row.
        settings_height = max(state["fit_config_frame"].winfo_reqheight() + 8,
                              geometry["preferred_height"] or 0)
        if settings.winfo_reqheight() != settings_height:
            settings.configure(height=settings_height)
        minimum = header.winfo_reqheight() + 14 + output.winfo_reqheight() + 8
        if not geometry["collapsed"]:
            minimum += settings_height
        height = max(minimum, workspace.canvas.winfo_height())
        if int(float(workspace.canvas.itemcget(workspace._window, "height"))) != height:
            workspace.canvas.itemconfigure(workspace._window, height=height)

    workspace_pending = {"id": None}

    def schedule_workspace(_event=None):
        if workspace_pending["id"] is None:
            workspace_pending["id"] = workspace.after_idle(fill_workspace)

    def schedule(_event=None):
        if (_event is not None and _event.type in (tk.EventType.Motion, tk.EventType.ButtonRelease)
                and not geometry["collapsed"]):
            geometry["dragged"] = True
        if geometry["pending"] is None:
            geometry["pending"] = split.after_idle(constrain)

    def toggle():
        if geometry["collapsed"]:
            geometry["collapsed"] = False
            geometry["preferred_height"] = geometry["saved_height"]
            fill_workspace()
            split.insert(0, settings, weight=0)
            split.sashpos(0, geometry["saved_height"])
            button.configure(text="▾ Fit settings")
            schedule()
        else:
            geometry["saved_height"] = split.sashpos(0)
            geometry["collapsed"] = True
            fill_workspace()
            split.forget(settings)
            button.configure(text="▸ Fit settings")
        schedule_workspace()

    button = tk.Button(header, text="▾ Fit settings", command=toggle,
                       anchor="w", padx=10, pady=7, relief="flat", borderwidth=0,
                       highlightthickness=1, font=ui_font_name(header, "ui_bold"))
    ui_native_colors(button, background="selected", foreground="text",
                     activebackground="pressed", activeforeground="text",
                     highlightbackground="selected", highlightcolor="accent")
    button.pack(side=tk.LEFT)
    summary_var = tk.StringVar(master=header)
    # Sibling widgets participate in normal geometry/paint order. A Label
    # placed inside a native Button can be overpainted during button redraw.
    summary = tk.Label(header, textvariable=summary_var, anchor="e",
                       font=ui_font_name(header, "ui"), borderwidth=0)
    ui_native_colors(summary, background="selected", foreground="text")
    summary.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 10))
    summary.bind("<Button-1>", lambda _event: button.invoke())
    header.bind("<Button-1>", lambda _event: button.invoke())
    button.bind("<Return>", lambda _event: button.invoke())

    def hover(active):
        color = ui_text_color(button, "pressed" if active else "selected")
        button.configure(background=color)
        summary.configure(background=color)
        header.configure(background=color)

    for widget in (header, button, summary):
        widget.bind("<Enter>", lambda _event: hover(True), add="+")
        widget.bind("<Leave>", lambda _event: hover(False), add="+")

    def update_summary(*_args):
        mode = state.get("fit_mode_var")
        value = mode.get() if mode is not None else "theta_alpha_multi"
        names = {"theta_alpha_multi": "Mode A · Ligand-donor",
                 "euler_global": "Mode B · Euler", "full_tensor": "Mode C · Full tensor + metal"}
        listing = state.get("fit_proton_list")
        count = len(listing.curselection()) if listing is not None else 0
        text = f"{names.get(value, value)} / {count} observation{'s' if count != 1 else ''}"
        if value == "theta_alpha_multi":
            donors = state.get("fit_donor_list")
            donor_count = len(donors.curselection()) if donors is not None else 0
            text += f" / {donor_count} donor{'s' if donor_count != 1 else ''}"
        summary_var.set(text)

    state["refresh_fit_settings_summary"] = update_summary
    state["fit_settings_summary_var"] = summary_var
    state["fit_settings_summary_label"] = summary
    state["fit_settings_toggle"] = button
    # Restore the sash during parent layout, before child Configure events
    # allocate an oversized intermediate Matplotlib bitmap.
    split.bind("<Configure>", lambda _event: constrain(), add="+")
    split.bind("<B1-Motion>", schedule, add="+")
    split.bind("<ButtonRelease-1>", schedule, add="+")
    workspace.canvas.bind("<Configure>", schedule_workspace, add="+")
    header.bind("<Configure>", schedule_workspace, add="+")
    output.bind("<Configure>", schedule_workspace, add="+")
    return settings, output


def build_fitting_result_split(parent, state):
    split = ttk.Panedwindow(parent, orient=tk.HORIZONTAL, style="PCS.TPanedwindow", height=280)
    split.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
    left = ttk.Frame(split, width=1, height=1)
    right = ttk.Frame(split, width=1, height=1)
    left.pack_propagate(False)
    right.pack_propagate(False)
    split.add(left, weight=1)
    split.add(right, weight=1)
    state["fit_result_split"] = split
    geometry = {"width": 0, "ratio": 0.5, "pending": None}

    def constrain():
        geometry["pending"] = None
        width = split.winfo_width()
        if width <= 1:
            return
        x = split.sashpos(0)
        if width != geometry["width"]:
            x = round(width * geometry["ratio"])
        minimum = min(180, max(30, (width - 8) // 3))
        x = max(minimum, min(width - minimum - 8, x))
        if split.sashpos(0) != x:
            split.sashpos(0, x)
        geometry.update(width=width, ratio=x / width)

    def schedule(_event=None):
        if geometry["pending"] is None:
            geometry["pending"] = split.after_idle(constrain)

    split.bind("<Configure>", schedule, add="+")
    split.bind("<B1-Motion>", schedule, add="+")
    split.bind("<ButtonRelease-1>", schedule, add="+")
    return left, right


def format_fit_conditions(result):
    """Read saved result metadata, never the current widget selection."""
    metadata = result.get("fit_metadata") or {}
    config = metadata.get("input_configuration") or {}
    mode = result.get("mode")
    labels = {"theta_alpha_multi": "Mode A · Ligand-donor",
              "euler_global": "Mode B · Euler",
              "full_tensor": "Mode C · Full tensor + metal"}
    rows = [("Model", labels.get(mode, str(mode)))]
    structured = config.get("observation_model") == "structured_v1"
    key = "observation_ids" if structured else "proton_ids"
    ids = config.get(key)
    if ids is None:
        # Old results still record which observations were actually fitted.
        observations = result.get("per_observable") or []
        if observations:
            ids = [row.get("observable_id") for row in observations]
            structured = True
        elif result.get("per_point"):
            ids = [row[0] for row in result["per_point"]]
    rows.append(("Observations", f"{len(ids)} used" if ids is not None else "Not recorded"))
    rows.append(("Observation IDs" if structured else "Ref IDs",
                 ", ".join(str(i) for i in ids) if ids else "Not recorded"))
    if mode == "theta_alpha_multi":
        donors = config.get("donor_ids", result.get("donor_ids"))
        rows.append(("Donor IDs", ", ".join(map(str, donors)) if donors else "Not recorded"))
        rows.append(("Axis mode", config.get("axis_mode", result.get("axis_mode", "Not recorded"))))
    if mode == "full_tensor":
        for label, parameter in (("Orientation", "orientation"), ("Metal position", "metal_pos")):
            status = (result.get("parameter_provenance") or {}).get(parameter, {}).get("status")
            rows.append((label, status.capitalize() if status else "Not recorded"))
    for label, parameter, flag in (("Δχ_ax", "delta_chi_ax", "fit_delta_chi_ax"),
                                   ("Δχ_rh", "delta_chi_rh", "fit_delta_chi_rh")):
        status = (result.get("parameter_provenance") or {}).get(parameter, {}).get("status")
        if not status and flag in config:
            status = "fitted" if config[flag] else "fixed"
        rows.append((label, status.capitalize() if status else "Not recorded"))
    weighting = metadata.get("weighting_mode", config.get("weighting_mode"))
    rows.append(("Weighting", {"unweighted": "Unweighted", "experimental_uncertainty":
                 "Experimental uncertainty"}.get(weighting, weighting or "Not recorded")))
    if mode in ("euler_global", "full_tensor"):
        initial = config.get("use_current_tensor")
        rows.append(("Initialization", "Not recorded" if initial is None else
                     "Current tensor" if initial else "Default initialization"))
    global_search = config.get("use_global_search")
    rows.append(("Global search", "Not recorded" if global_search is None else
                 "On" if global_search else "Off"))
    optimizer = metadata.get("optimizer")
    rows.append(("Optimizer", {"least_squares": "Least squares",
                 "differential_evolution_then_least_squares": "Global search + least squares"}
                 .get(optimizer, optimizer or "Not recorded")))
    lines = ["Fit conditions — recorded at run time", "------------------------------------"]
    for label, value in rows:
        lines.extend(textwrap.wrap(str(value), width=64, initial_indent=f"{label:<15}: ",
                                   subsequent_indent=" " * 17, break_long_words=False,
                                   break_on_hyphens=False) or [f"{label:<15}: -"])
    return "\n".join(lines) + "\n\n"


def show_fit_empty_guide(figure, state):
    fonts = state.get("fonts", {}) or {}
    figure.text(0.5, 0.60, "Fitting", ha="center", va="center",
                fontsize=fonts.get("plot_title", 10), fontweight="bold")
    figure.text(0.5, 0.43,
                "Select a model and PCS observations.\nThen click Run Fit.",
                ha="center", va="center", wrap=True, linespacing=1.6,
                fontsize=fonts.get("plot_label", 9), alpha=0.72)
