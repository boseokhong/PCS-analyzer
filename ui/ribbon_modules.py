"""Tools Ribbon module launchers and the ribbon-only ordering dialog."""
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

from logic.app_settings import save_app_state
from plugin_system.ribbon_modules import category_icon, loaded_launchers, normalize_order, ordered_launchers
from ui.ribbon.ribbon_button import RibbonButton
from ui.style import get_app_fonts
from ui.tooltips import ToolTip
from ui.commands import Command
from ui.ribbon.icon_service import tk_scale_factor


def elide_label(text, measure, available):
    """Fit display text only; retain the original launcher label and identity."""
    text = " ".join(str(text).split())
    if measure(text) <= available:
        return text
    suffix = "…"
    if measure(suffix) > available:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(text[:middle] + suffix) <= available:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + suffix


class RibbonModules:
    def __init__(self, state, group, manager_button):
        self.state, self.group = state, group
        self.tools = []
        self.buttons = {}
        self.window = None
        self._fit_after = None
        content = group.content
        content.bind("<Configure>", self.schedule_label_fit, add="+")
        content.bind("<<ThemeChanged>>", self.schedule_label_fit, add="+")
        content.bind("<Destroy>", self.cancel_label_fit, add="+")
        manager_button.grid(row=0, column=0, sticky="ew", padx=1, pady=(0, 1))
        self.arrange = RibbonButton(content, text="Arrange modules…", compound="left",
                                    image=state["ribbon_icons"].resolve("settings"), command=self.open_dialog)
        self.arrange.grid(row=1, column=0, sticky="ew", padx=1, pady=(0, 1))
        ToolTip(self.arrange, "Change the order of modules in the Tools ribbon.")
        self.divider = ttk.Separator(content, orient="vertical", style="Ribbon.Separator.TSeparator")
        self.more = RibbonButton(content, text="More ▾", command=self.open_more, compound="left")
        self.menu = tk.Menu(self.more, tearoff=False)
        state["refresh_ribbon_modules"] = self.refresh
        group.overflow_commands_supplier = self.overflow_commands
        state["ribbon"].bind("<<RibbonPageChanged>>", self.refresh_enabled, add="+")

    def refresh_enabled(self, _event=None):
        by_id = {tool.id: tool for tool in loaded_launchers(self.state)}
        for key, button in self.buttons.items():
            current = by_id.get(key)
            button.state(["!disabled"] if current is not None and current.is_enabled() else ["disabled"])

    def overflow_commands(self):
        return [self.state["stable_ribbon_buttons"]["app.tools.module_manager"].command_model,
                Command("ribbon.modules.arrange", "Arrange modules…", self.open_dialog,
                        enabled=bool(self.tools))] + [
                    Command(f"ribbon.module.{tool.id}", tool.label,
                            lambda t=tool: self.invoke(t), enabled=tool.is_enabled)
                    for tool in self.tools
                ]

    def refresh(self):
        self.tools = loaded_launchers(self.state)
        settings = self.state["app_settings"]
        saved = normalize_order(settings.get("ribbon_module_order"))
        self.tools = ordered_launchers(self.tools, saved)
        if saved:
            extended = saved + [tool.id for tool in self.tools if tool.id not in saved]
            if extended != saved:
                settings["ribbon_module_order"] = extended
                save_app_state(self.state, save_geometry=False)
        visible = {tool.id for tool in self.tools[:3]}
        for key in tuple(self.buttons):
            if key not in visible:
                self.buttons.pop(key).destroy()
        for row, tool in enumerate(self.tools[:3]):
            button = self.buttons.get(tool.id)
            if button is None:
                button = self.buttons[tool.id] = RibbonButton(self.group.content, compound="left")
                button.bind("<Enter>", self.refresh_enabled, add="+")
            button.configure(text=tool.label, width=0, command=lambda t=tool: self.invoke(t),
                             state="normal" if tool.is_enabled() else "disabled")
            button.set_image(self.state["ribbon_icons"].resolve(category_icon(tool.category)))
            button._tooltip.update_text(tool.label + (f"\n{tool.tooltip}" if tool.tooltip and tool.tooltip != tool.label else ""))
            button.grid(row=row, column=2, sticky="ew", padx=3, pady=(0, 1))
        self.arrange.state(["!disabled"] if self.tools else ["disabled"])
        if self.tools:
            self.divider.grid(row=0, column=1, rowspan=3, sticky="ns", padx=5, pady=3)
        else:
            self.divider.grid_remove()
        if len(self.tools) > 3:
            self.more.grid(row=0, column=3, sticky="ew", padx=3)
        else:
            self.more.grid_remove()
        if self.window is not None and self.window.winfo_exists():
            self.reconcile_dialog()
        self.fit_labels()

    def schedule_label_fit(self, _event=None):
        if self._fit_after is None:
            self._fit_after = self.group.content.after_idle(self.fit_labels)

    def cancel_label_fit(self, event):
        if event.widget is self.group.content and self._fit_after is not None:
            self.group.content.after_cancel(self._fit_after)
            self._fit_after = None

    def fit_labels(self):
        if self._fit_after is not None:
            self.group.content.after_cancel(self._fit_after)
            self._fit_after = None
        root = self.state["root"]
        style = ttk.Style(root)
        limit = max(1, round(240 * tk_scale_factor(root)))
        tools = {tool.id: tool for tool in self.tools}
        for key, button in self.buttons.items():
            font = tkfont.Font(root=root, font=style.lookup(button.cget("style"), "font"))
            image = button._ribbon_image
            image_width = int(root.tk.call("image", "width", image)) if image not in (None, "") else 0
            # Include shared compact padding, border/focus insets and compound gap.
            padding = root.tk.splitlist(style.lookup(button.cget("style"), "padding") or "0")
            horizontal = root.winfo_pixels(padding[0]) + root.winfo_pixels(padding[2] if len(padding) > 2 else padding[0])
            overhead = image_width + horizontal + round(12 * tk_scale_factor(root))
            label = elide_label(tools[key].label, font.measure, max(0, limit - overhead))
            if str(button.cget("text")) != label:
                button.configure(text=label)

    def invoke(self, tool):
        # Resolve again: an open menu must not launch a removed/disabled module.
        current = next((entry for entry in loaded_launchers(self.state) if entry.id == tool.id), None)
        if current is not None and current.is_enabled():
            return current.launch()

    def open_more(self):
        self.menu.delete(0, "end")
        for tool in self.tools[3:]:
            self.menu.add_command(label=tool.label, command=lambda t=tool: self.invoke(t),
                                  image=self.state["ribbon_icons"].resolve(category_icon(tool.category)), compound="left",
                                  state="normal" if tool.is_enabled() else "disabled")
        try:
            self.menu.tk_popup(self.more.winfo_rootx(), self.more.winfo_rooty() + self.more.winfo_height())
        finally:
            self.menu.grab_release()

    def open_dialog(self):
        if self.window is not None and self.window.winfo_exists():
            self.window.lift()
            return self.window
        root = self.state["root"]
        win = self.window = tk.Toplevel(root)
        win.withdraw()
        win.title("Arrange ribbon modules")
        win.transient(root)
        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Module order", font=get_app_fonts(self.state)["ui_bold"]).pack(anchor="w", pady=(0, 6))
        area = ttk.Frame(body, padding=0)
        area.pack(fill="both", expand=True)
        self.table = ttk.Treeview(area, columns=("location",), show="tree headings", selectmode="browse", height=8)
        self.table.heading("#0", text="Module")
        self.table.heading("location", text="Location")
        self.table.column("#0", width=300, minwidth=150)
        self.table.column("location", width=75, minwidth=65, stretch=False)
        scroll = ttk.Scrollbar(area, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        arrows = ttk.Frame(area, padding=(8, 0, 0, 0))
        arrows.pack(side="left", fill="y")
        self.up = ttk.Button(arrows, text="↑", width=3, command=lambda: self.move(-1))
        self.down = ttk.Button(arrows, text="↓", width=3, command=lambda: self.move(1))
        self.up.pack(pady=(35, 4))
        self.down.pack()
        ToolTip(self.up, "Move up")
        ToolTip(self.down, "Move down")
        self.table.bind("<<TreeviewSelect>>", self.update_arrows)
        self.table.bind("<ButtonPress-1>", self.drag_start, add="+")
        self.table.bind("<ButtonRelease-1>", self.drag_end, add="+")
        ttk.Label(body, text="Drag to reorder, or select a module and use ↑ / ↓.", wraplength=420).pack(anchor="w", pady=(10, 3))
        ttk.Label(body, text="First 3 modules appear directly in the ribbon; the rest appear under More.", wraplength=420).pack(anchor="w")
        ttk.Separator(body).pack(fill="x", pady=12)
        footer = ttk.Frame(body, padding=0)
        footer.pack(fill="x")
        ttk.Button(footer, text="Restore default", command=self.restore_default).pack(side="left")
        ttk.Button(footer, text="Cancel", command=self.close_dialog).pack(side="right")
        ttk.Button(footer, text="Apply", style="Primary.TButton", command=self.apply).pack(side="right", padx=(0, 6))
        self.draft = [tool.id for tool in self.tools]
        self.render_dialog()
        win.protocol("WM_DELETE_WINDOW", self.close_dialog)
        win.bind("<Escape>", lambda event: self.close_dialog())
        win.update_idletasks()
        width, height = win.winfo_reqwidth(), win.winfo_reqheight()
        x = root.winfo_rootx() + (root.winfo_width() - width) // 2
        y = root.winfo_rooty() + (root.winfo_height() - height) // 2
        win.geometry(f"{width}x{height}{x:+d}{y:+d}")
        win.minsize(360, min(height, 300))
        win.deiconify()
        return win

    def close_dialog(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def reconcile_dialog(self):
        self.draft = [tool.id for tool in ordered_launchers(self.tools, self.draft)]
        self.render_dialog()

    def render_dialog(self, selected=None):
        selected = selected or next(iter(self.table.selection()), None)
        self.table.delete(*self.table.get_children())
        by_id = {tool.id: tool for tool in self.tools}
        for index, key in enumerate(self.draft):
            self.table.insert("", "end", iid=key, text=by_id[key].label,
                              values=("Ribbon" if index < 3 else "More",))
        if self.draft:
            key = selected if selected in self.draft else self.draft[0]
            self.table.selection_set(key)
            self.table.focus(key)
            self.table.see(key)
        self.update_arrows()

    def update_arrows(self, _event=None):
        selection = self.table.selection()
        index = self.draft.index(selection[0]) if selection and selection[0] in self.draft else -1
        self.up.state(["!disabled"] if index > 0 else ["disabled"])
        self.down.state(["!disabled"] if 0 <= index < len(self.draft) - 1 else ["disabled"])

    def move(self, delta):
        selection = self.table.selection()
        if not selection:
            return
        key = selection[0]
        index = self.draft.index(key)
        target = index + delta
        if 0 <= target < len(self.draft):
            self.draft[index], self.draft[target] = self.draft[target], self.draft[index]
            self.render_dialog(key)

    def drag_start(self, event):
        self.dragged = self.table.identify_row(event.y)

    def drag_end(self, event):
        target = self.table.identify_row(event.y)
        key = getattr(self, "dragged", None)
        self.dragged = None
        if key in self.draft and target in self.draft and key != target:
            target_index = self.draft.index(target)
            self.draft.remove(key)
            self.draft.insert(target_index, key)
            self.render_dialog(key)

    def restore_default(self):
        self.draft = [tool.id for tool in ordered_launchers(self.tools, [])]
        self.render_dialog()

    def apply(self):
        # Keep unavailable identities for reinstalls; they consume no ribbon slots.
        saved = normalize_order(self.state["app_settings"].get("ribbon_module_order"))
        self.state["app_settings"]["ribbon_module_order"] = self.draft + [key for key in saved if key not in self.draft]
        save_app_state(self.state, save_geometry=False)
        self.refresh()
