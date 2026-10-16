"""Compact presentation-only tabs for the PCS document manager."""

from __future__ import annotations

from collections import Counter
from ui.theme_palette import document_palette, document_arrow_palette, theme_id

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from ui.style import get_app_fonts


DocumentCallback = Callable[[str], Any]


class DocumentTabStrip(ttk.Frame):
    """Render manager documents without owning any document state."""

    def __init__(
        self,
        master,
        manager,
        *,
        on_new: Callable[[], Any] | None = None,
        on_close: DocumentCallback | None = None,
        on_save: DocumentCallback | None = None,
        on_save_as: DocumentCallback | None = None,
        on_duplicate: DocumentCallback | None = None,
        on_project_name: DocumentCallback | None = None,
        on_tab_color: Callable[[str, str | None], Any] | None = None,
        on_close_others: DocumentCallback | None = None,
        on_close_all: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(master, padding=0)
        self.manager = manager
        self._on_new = on_new or manager.new_document
        self._on_close = on_close or manager.close_document
        self._on_save = on_save
        self._on_save_as = on_save_as
        self._on_duplicate = on_duplicate
        self._on_project_name = on_project_name
        self._on_tab_color = on_tab_color
        self._on_close_others = on_close_others
        self._on_close_all = on_close_all
        self._listener_token = manager.subscribe(self._active_document_changed)
        self._tab_widgets: dict[
            str, tuple[tk.Canvas, tk.Frame, tk.Button, tk.Button]
        ] = {}
        self._rendered_document_ids: tuple[str, ...] = ()
        self._context_document_id: str | None = None
        self._drag_document_id: str | None = None

        root = self._root()
        self._fonts = get_app_fonts(root)
        dark = str(getattr(root, "_theme_variant", "light")).lower() == "dark"
        self._palette = document_palette(theme_id(root))
        self._palette.update(
            strip=getattr(root, "_app_bg", self._palette["strip"]),
            active=getattr(root, "_app_palette", {}).get("field", self._palette["active"]),
            active_hover=getattr(root, "_app_palette", {}).get("field", self._palette["active_hover"]),
            foreground=getattr(root, "_app_fg", self._palette["foreground"]),
        )

        self._row = tk.Frame(
            self, bg=self._palette["strip"], bd=0, highlightthickness=0
        )
        self._row.pack(fill=tk.X)
        self._row.grid_columnconfigure(0, weight=1)

        self._navigation = tk.Frame(self._row, width=45, height=25,
                                    bg=self._palette["strip"], bd=0, highlightthickness=0)
        self._navigation.grid(row=0, column=1, sticky="se", padx=(5, 0))
        self._navigation.grid_remove()
        self._arrow_frames: dict[int, tk.Frame] = {}
        self._scroll_left = self._scroll_button("‹", -1)
        self._canvas = tk.Canvas(
            self._row,
            height=28,
            bd=0,
            highlightthickness=0,
            bg=self._palette["strip"],
            xscrollincrement=24,
        )
        self._canvas.grid(row=0, column=0, sticky="ew")
        self._tabs_host = tk.Frame(
            self._canvas,
            bg=self._palette["strip"],
            bd=0,
            highlightthickness=0,
        )
        self._host_window = self._canvas.create_window(
            0, 0, anchor="nw", window=self._tabs_host
        )
        self._scroll_right = self._scroll_button("›", 1)

        self._new_slot = tk.Canvas(
            self._tabs_host, width=25, height=28, bd=0, highlightthickness=0
        )
        self._new_slot.pack(side=tk.LEFT, anchor=tk.S)
        self._new_button = tk.Button(
            self._new_slot,
            text="+",
            command=self.new_document,
            padx=7,
            pady=1,
            relief=tk.FLAT,
            bd=0,
            bg=self._palette["inactive"],
            activebackground=self._palette["inactive_hover"],
            fg=self._palette["foreground"],
            activeforeground=self._palette["foreground"],
            font=self._fonts["ui_small"],
            cursor="hand2",
            takefocus=True,
        )

        self._bottom_border = tk.Canvas(
            self,
            height=1,
            bd=0,
            highlightthickness=0,
            bg=self._palette["active"],
        )
        # The rail shares the last row of the tabs instead of adding a row.
        self._bottom_border.place(x=0, rely=1, y=-1, relwidth=1, height=1)

        self._context_menu = tk.Menu(self, tearoff=False)
        self._add_context_command("Save", self._on_save)
        self._add_context_command("Save As...", self._on_save_as)
        self._context_menu.add_separator()
        self._add_context_command("Duplicate Document", self._on_duplicate)
        self._add_context_command("Project Name...", self._on_project_name)
        color_menu = tk.Menu(self._context_menu, tearoff=False)
        for label, color in (
            ("Blue", "#B7D7F5"), ("Green", "#BFE3C0"),
            ("Amber", "#F2D49B"), ("Rose", "#E8B8C5"),
            ("Violet", "#D1C1EA"), ("Default", None),
        ):
            color_menu.add_command(
                label=label,
                command=lambda value=color: self._invoke_tab_color(value),
                state="normal" if callable(self._on_tab_color) else "disabled",
            )
        self._context_menu.add_cascade(label="Tab Color", menu=color_menu)
        self._context_menu.add_separator()
        self._add_context_command("Close", self._on_close)
        self._add_context_command("Close Others", self._on_close_others)
        self._context_menu.add_command(
            label="Close All",
            command=lambda: self._invoke_all(self._on_close_all),
            state="normal" if callable(self._on_close_all) else "disabled",
        )

        self._tabs_host.bind("<Configure>", self._on_host_configure, add="+")
        self._canvas.bind("<Configure>", self._on_canvas_configure, add="+")
        self._canvas.bind(
            "<Shift-MouseWheel>",
            lambda event: self._canvas.xview_scroll(
                -1 if event.delta > 0 else 1, "units"
            ),
            add="+",
        )
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._canvas.configure(xscrollcommand=self._on_xscroll)
        self.refresh()

    def _scroll_button(self, text: str, direction: int) -> tk.Button:
        # A real outer surface owns the permanent border; native Button focus
        # highlights alone do not reliably render the requested square outline.
        border = tk.Frame(self._navigation, bd=0, highlightthickness=0)
        self._arrow_frames[direction] = border
        button = tk.Button(
            border,
            text=text,
            command=lambda: self._canvas.xview_scroll(direction * 4, "units"),
            padx=0,
            pady=0,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightcolor=getattr(self._root(), "_app_palette", {}).get("accent", "#356f9f"),
            bg=self._palette["inactive"],
            activebackground=self._palette["inactive_hover"],
            fg=self._palette["foreground"],
            activeforeground=self._palette["foreground"],
            font=self._fonts["ui_small"],
            anchor=tk.CENTER,
            takefocus=True,
        )
        button.place(x=1, y=1, relwidth=1, width=-2, relheight=1, height=-2)
        return button

    def _on_xscroll(self, first, last) -> None:
        dark = getattr(self._root(), "_theme_variant", "light") == "dark"
        colors = document_arrow_palette(theme_id(self._root()))
        for button, disabled in ((self._scroll_left, float(first) <= 0.001),
                                 (self._scroll_right, float(last) >= 0.999)):
            background = colors["disabled_background" if disabled else "background"]
            edge = colors["disabled_edge" if disabled else "edge"]
            button.master.configure(bg=edge)
            button.configure(
                state=tk.DISABLED if disabled else tk.NORMAL,
                bg=background,
                fg=colors["foreground"],
                activebackground=colors["hover"],
                activeforeground=colors["hover_foreground"],
                disabledforeground=colors["disabled_foreground"],
                highlightbackground=background,
                highlightcolor=getattr(self._root(), "_app_palette", {}).get("accent", "#356f9f"),
            )
        self.after_idle(self._draw_bottom_border)

    def _add_context_command(
        self, label: str, callback: DocumentCallback | None
    ) -> None:
        self._context_menu.add_command(
            label=label,
            command=lambda action=callback: self._invoke_target(action),
            state="normal" if callable(callback) else "disabled",
        )

    @staticmethod
    def display_titles(documents) -> dict[str, str]:
        documents = tuple(documents)
        base_titles = {
            document.document_id: (document.project_name or document.title)
            for document in documents
        }
        counts = Counter(base_titles.values())
        labels: dict[str, str] = {}
        for document in documents:
            title = base_titles[document.document_id]
            if counts[title] > 1 and document.file_path is not None:
                parent = document.file_path.parent.name
                if parent:
                    title = f"{title} — {parent}"
            labels[document.document_id] = title
        return labels

    @property
    def tab_document_ids(self) -> tuple[str, ...]:
        return self._rendered_document_ids

    @property
    def active_document_id(self) -> str:
        return self.manager.active_document_id

    @property
    def add_button_follows_tabs(self) -> bool:
        return self._new_slot.master is self._tabs_host

    def tab_label(self, document_id: str) -> str | None:
        widgets = self._tab_widgets.get(str(document_id))
        return str(widgets[2].cget("text")) if widgets else None

    def activate_document(self, document_id: str):
        document = self.manager.set_active_document(str(document_id))
        self.refresh()
        return document

    def new_document(self):
        document = self._on_new()
        self.refresh()
        self.after_idle(lambda: self._canvas.xview_moveto(1.0))
        return document

    def close_document(self, document_id: str):
        closed = self._on_close(str(document_id))
        self.refresh()
        return closed

    def reorder_document(self, document_id: str, new_index: int) -> None:
        self.manager.move_document(str(document_id), int(new_index))
        self.refresh()

    def _invoke_target(self, callback: DocumentCallback | None):
        if callable(callback) and self._context_document_id is not None:
            result = callback(self._context_document_id)
            self.refresh()
            return result
        return None

    def _invoke_all(self, callback: Callable[[], Any] | None):
        if callable(callback):
            result = callback()
            self.refresh()
            return result
        return None

    def _invoke_tab_color(self, color: str | None):
        if callable(self._on_tab_color) and self._context_document_id is not None:
            result = self._on_tab_color(self._context_document_id, color)
            self.refresh()
            return result
        return None

    def _show_context_menu(self, document_id: str, event) -> str:
        self._context_document_id = str(document_id)
        try:
            self._context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._context_menu.grab_release()
        return "break"

    def _start_drag(self, document_id: str) -> None:
        self._drag_document_id = str(document_id)

    def _finish_drag(self, event) -> None:
        document_id = self._drag_document_id
        self._drag_document_id = None
        if document_id is None or document_id not in self._tab_widgets:
            return
        target_index = 0
        for other_id in self._rendered_document_ids:
            if other_id == document_id:
                continue
            outer = self._tab_widgets[other_id][0]
            midpoint = outer.winfo_rootx() + outer.winfo_width() / 2
            if event.x_root > midpoint:
                target_index += 1
        self.reorder_document(document_id, target_index)

    def _bind_tab_widget(self, widget, document_id: str) -> None:
        widget.bind(
            "<ButtonPress-1>",
            lambda _event, value=document_id: self._start_drag(value),
            add="+",
        )
        widget.bind("<ButtonRelease-1>", self._finish_drag, add="+")
        widget.bind(
            "<Button-2>",
            lambda _event, value=document_id: self.close_document(value),
            add="+",
        )
        widget.bind(
            "<Button-3>",
            lambda event, value=document_id: self._show_context_menu(
                value, event
            ),
            add="+",
        )

    def _active_document_changed(self, _event) -> None:
        try:
            self.after_idle(self.refresh)
        except tk.TclError:
            pass

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        try:
            self.manager.unsubscribe(self._listener_token)
        except Exception:
            pass

    def _on_host_configure(self, _event=None) -> None:
        # refresh() owns the row height; transient host geometry must not resize
        # the entire workspace while tabs are being rebuilt.
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._update_overflow()

    def _on_canvas_configure(self, _event=None) -> None:
        self._update_overflow()
        self.after_idle(self._draw_bottom_border)

    def _update_overflow(self) -> None:
        try:
            overflowing = (
                self._tabs_host.winfo_reqwidth() > self._row.winfo_width()
            )
        except tk.TclError:
            return
        if overflowing:
            self._navigation.grid()
        else:
            self._navigation.grid_remove()
            self._canvas.xview_moveto(0.0)

    def _draw_bottom_border(self) -> None:
        try:
            width = self._bottom_border.winfo_width()
            self._bottom_border.delete("all")
            self._bottom_border.create_rectangle(0, 0, width, 1, outline="", fill=self._palette["border"])
            active = self._tab_widgets.get(self.manager.active_document_id)
            if active is not None:
                start = active[0].winfo_rootx() - self.winfo_rootx()
                end = start + active[0].winfo_width()
                viewport_start = self._canvas.winfo_rootx() - self.winfo_rootx()
                start = max(start, viewport_start)
                end = min(end, viewport_start + self._canvas.winfo_width())
                accent = getattr(self._root(), "_app_palette", {}).get("accent", "#356f9f")
                if end > start:
                    self._bottom_border.create_rectangle(start, 0, end, 1, outline="", fill=accent)
        except tk.TclError:
            pass

    def _ensure_active_visible(self) -> None:
        try:
            active = self._tab_widgets.get(self.manager.active_document_id)
            if active is None:
                return
            outer = active[0]
            total = max(1, self._tabs_host.winfo_reqwidth())
            viewport = self._canvas.winfo_width()
            left = outer.winfo_x()
            right = left + outer.winfo_width()
            visible_left = self._canvas.canvasx(0)
            visible_right = visible_left + viewport
            if left < visible_left:
                self._canvas.xview_moveto(left / total)
            elif right > visible_right:
                self._canvas.xview_moveto(max(0.0, (right - viewport) / total))
        except tk.TclError:
            pass

    def _rebuild_tabs(self, document_ids: tuple[str, ...]) -> None:
        for child in self._tabs_host.winfo_children():
            if child is not self._new_slot:
                child.destroy()
        self._tab_widgets.clear()

        for document_id in document_ids:
            outer = tk.Canvas(
                self._tabs_host,
                width=1,
                height=int(self._canvas.cget("height")),
                bg=self._palette["border"],
                bd=0,
                highlightthickness=0,
            )
            outer.pack(
                side=tk.LEFT,
                anchor=tk.S,
                padx=(0, 1),
                before=self._new_slot,
            )
            content = tk.Frame(outer, bd=0, highlightthickness=0)
            title = tk.Button(
                content,
                command=lambda value=document_id: self.activate_document(value),
                padx=7,
                pady=0,
                relief=tk.FLAT,
                bd=0,
                cursor="hand2",
                takefocus=True,
            )
            title.pack(side=tk.LEFT, fill=tk.Y)
            close = tk.Button(
                content,
                text="×",
                command=lambda value=document_id: self.close_document(value),
                padx=3,
                pady=0,
                relief=tk.FLAT,
                bd=0,
                cursor="hand2",
                takefocus=True,
            )
            close.pack(side=tk.LEFT, fill=tk.Y)
            self._tab_widgets[document_id] = (outer, content, title, close)
            for widget in (outer, content, title):
                self._bind_tab_widget(widget, document_id)
        self._rendered_document_ids = document_ids

    def refresh(self) -> None:
        try:
            documents = tuple(self.manager.documents.values())
        except (AttributeError, tk.TclError):
            return
        document_ids = tuple(document.document_id for document in documents)
        if document_ids != self._rendered_document_ids:
            self._rebuild_tabs(document_ids)

        labels = self.display_titles(documents)
        active_id = self.manager.active_document_id
        for document in documents:
            outer, content, title, close = self._tab_widgets[
                document.document_id
            ]
            active = document.document_id == active_id
            background = document.tab_color or self._palette["active" if active else "inactive"]
            hover = self._palette["active_hover" if active else "inactive_hover"]
            foreground = self._palette["foreground" if active else "inactive_foreground"]
            if document.tab_color:
                try:
                    red, green, blue = (
                        int(document.tab_color[index:index + 2], 16)
                        for index in (1, 3, 5)
                    )
                    foreground = "#111111" if (red * 299 + green * 587 + blue * 114) >= 128000 else "#ffffff"
                except (TypeError, ValueError):
                    pass
            content.configure(bg=background)
            label = labels[document.document_id]
            title.configure(
                text=f"{label}{' *' if document.dirty else ''}",
                bg=background,
                activebackground=hover,
                fg=foreground,
                activeforeground=foreground,
                font=self._fonts["ui_small"],
            )
            close.configure(
                bg=background,
                activebackground=hover,
                fg=foreground,
                activeforeground=foreground,
                font=self._fonts["ui_small"],
            )
        # Button requested sizes are available after configure(). Let Tk process
        # layout only once all faces have their final dimensions, rather than
        # flushing an intermediate rebuild across the whole application.
        buttons = [self._new_button]
        for _, _, title, close in self._tab_widgets.values():
            buttons.extend((title, close))
        height = max(28, max(button.winfo_reqheight() for button in buttons) + 8)
        arrow_width = max(22, self._scroll_left.winfo_reqwidth() + 2, self._scroll_right.winfo_reqwidth() + 2)
        self._navigation.configure(width=arrow_width * 2 + 1, height=height - 3)
        self._arrow_frames[-1].place(x=0, y=0, width=arrow_width, height=height - 3)
        self._arrow_frames[1].place(x=arrow_width + 1, y=0, width=arrow_width, height=height - 3)
        for document in documents:
            outer, content, title, close = self._tab_widgets[document.document_id]
            active = document.document_id == active_id
            width = title.winfo_reqwidth() + close.winfo_reqwidth() + 10
            top = 0 if active else 3
            self._paint_face(outer, width, height, top, content.cget("bg"), active)
            content.place(x=5, y=top + 1, width=width - 10, height=height - top - (3 if active else 2))
        new_width = max(25, self._new_button.winfo_reqwidth() + 2)
        self._paint_face(self._new_slot, new_width, height, 3, self._palette["inactive"], False)
        self._new_button.place(x=1, y=4, width=new_width - 2, height=height - 5)
        self._canvas.configure(height=height)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._update_overflow()
        self.after_idle(self._ensure_active_visible)
        self.after_idle(self._draw_bottom_border)

    def _paint_face(self, canvas, width, height, top, background, selected):
        """Draw the whole tab face once, inside a fixed-height slot."""
        canvas.configure(width=width, height=height, background=self._palette["strip"])
        canvas.delete("all")
        canvas.create_rectangle(0, top, width - 1, height - 1, fill=background,
                                outline=self._palette["border"], tags="face")
        if selected:
            accent = getattr(self._root(), "_app_palette", {}).get("accent", "#356f9f")
            canvas.create_rectangle(0, height - 2, width, height, fill=accent,
                                    outline="", tags="selection")

    def refresh_styles(self) -> None:
        """Refresh existing chrome without changing document/tab ownership."""
        root = self._root()
        dark = getattr(root, "_theme_variant", "light") == "dark"
        self._fonts = get_app_fonts(root)
        colors = document_palette(theme_id(root))
        self._palette.update(
            **{key: colors[key] for key in ("inactive", "inactive_hover", "inactive_foreground", "border")},
            strip=getattr(root, "_app_bg", self._palette["strip"]),
            active=getattr(root, "_app_palette", {}).get("field", colors["active"]),
            active_hover=getattr(root, "_app_palette", {}).get("field", colors["active_hover"]),
            foreground=getattr(root, "_app_fg", self._palette["foreground"]),
        )
        for widget in (self._row, self._canvas, self._tabs_host, self._navigation):
            widget.configure(background=self._palette["strip"])
        self._bottom_border.configure(background=self._palette["active"])
        for button in (self._new_button, self._scroll_left, self._scroll_right):
            button.configure(background=self._palette["inactive"],
                             activebackground=self._palette["inactive_hover"],
                             foreground=self._palette["foreground"],
                             activeforeground=self._palette["foreground"],
                             font=self._fonts["ui_small"])
        self.refresh()
        self._on_xscroll(*self._canvas.xview())


__all__ = ["DocumentTabStrip"]
