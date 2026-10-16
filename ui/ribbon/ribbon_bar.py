"""Compact tabbed Ribbon shell used above the PCS Analyzer main content."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import tkinter as tk
from tkinter import ttk

from .ribbon_page import RibbonPage
from .styles import configure_ribbon_styles
from .icon_service import tk_scale_factor
from ui.tooltips import ToolTip


class RibbonTabStrip(ttk.Frame):
    """Tab buttons and collapse control for a :class:`RibbonBar`."""

    def __init__(self, master, *, on_select, on_toggle_collapsed) -> None:
        super().__init__(master, style="Ribbon.TabStrip.TFrame", padding=(4, 0, 4, 0))
        self._on_select = on_select
        self._on_toggle_collapsed = on_toggle_collapsed
        self._buttons: dict[str, ttk.Button] = {}
        self._tab_frames: dict[str, ttk.Frame] = {}
        self._underlines: dict[str, ttk.Frame] = {}
        self._contextual: set[str] = set()
        self._selected_page_id: str | None = None
        self._hovered_page_id: str | None = None
        self._label_candidates: dict[str, tuple[str, ...]] = {}
        self._reserved_sizes: dict[str, tuple[int, int]] = {}
        self._label_measure: ttk.Button | None = None

        self.tabs = ttk.Frame(self, style="Ribbon.TabStrip.TFrame")
        self.collapse_button = ttk.Button(
            self,
            text="▴",
            width=3,
            command=on_toggle_collapsed,
            style="Ribbon.Collapse.TButton",
            takefocus=True,
        )
        self.collapse_button.pack(side="right", padx=(4, 0))
        self.tabs.pack(side="left", fill="x", expand=True)
        self._bind_tab_wheel(self)
        self._bind_tab_wheel(self.tabs)
        self._collapse_tooltip = ToolTip(
            self.collapse_button,
            "Collapse or expand the Ribbon. You can also double-click any Ribbon tab.",
        )

    def add_tab(self, page_id: str, label: str, *, contextual: bool = False) -> None:
        if page_id in self._buttons:
            raise ValueError(f"Ribbon tab already exists: {page_id}")

        container = ttk.Frame(
            self.tabs,
            style=(
                "Ribbon.ContextTab.Container.TFrame"
                if contextual
                else "Ribbon.Tab.Container.TFrame"
            ),
        )
        container.pack(side="left", padx=(6, 1) if contextual else (0, 1))
        button = ttk.Button(
            container,
            text=label,
            command=lambda key=page_id: self._on_select(key),
            style="Ribbon.ContextTab.TButton" if contextual else "Ribbon.Tab.TButton",
            takefocus=True,
        )
        button.pack(side="top", fill="x")
        # A double click toggles the Ribbon without changing ordinary single-click
        # page selection semantics. Returning "break" suppresses duplicate bind work.
        button.bind("<Double-Button-1>", self._toggle_from_tab, add="+")

        underline = ttk.Frame(
            container,
            height=2,
            style=(
                "Ribbon.ContextTab.Underline.TFrame"
                if contextual
                else "Ribbon.Tab.Underline.TFrame"
            ),
        )
        underline.pack(side="bottom", fill="x")
        underline.pack_propagate(False)

        self._buttons[page_id] = button
        self._tab_frames[page_id] = container
        self._underlines[page_id] = underline
        for widget in (container, button, underline):
            self._bind_tab_wheel(widget)
            widget.bind("<Enter>", lambda e, key=page_id: self._set_hover(key, True), add="+")
            widget.bind("<Leave>", lambda e, key=page_id: self._set_hover(key, False), add="+")
        if contextual:
            self._contextual.add(page_id)
            self.reserve_label_space(page_id, (label,))

    def _bind_tab_wheel(self, widget) -> None:
        # Local bindings only: command pages and other tab systems retain their
        # own scrolling/selection behavior.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self._wheel_select_tab, add="+")

    def _wheel_select_tab(self, event):
        delta = getattr(event, "delta", 0)
        number = getattr(event, "num", None)
        if number == 4:
            direction, steps = -1, 1
        elif number == 5:
            direction, steps = 1, 1
        elif delta:
            direction = -1 if delta > 0 else 1
            steps = max(1, abs(int(delta)) // 120)
        else:
            return "break"
        pages = [key for key, button in self._buttons.items()
                 if self._tab_frames[key].winfo_ismapped()
                 and not button.instate(["disabled"])]
        if self._selected_page_id in pages:
            current = pages.index(self._selected_page_id)
            target = max(0, min(len(pages) - 1, current + direction * steps))
            if target != current:
                self._on_select(pages[target])
        return "break"

    def _toggle_from_tab(self, _event=None):
        self._on_toggle_collapsed()
        return "break"

    def set_label(self, page_id: str, label: str) -> None:
        label = str(label)
        button = self._buttons[page_id]
        if str(button.cget("text")) == label:
            return
        if page_id in self._contextual and label not in self._label_candidates[page_id]:
            self.reserve_label_space(page_id, self._label_candidates[page_id] + (label,))
        button.configure(text=label)

    def reserve_label_space(self, page_id: str, labels, *, force: bool = False) -> None:
        """Measure context labels with native ttk metrics, without flushing Tk.

        One hidden, command-free measurement widget covers both selected styles.
        The actual button, its background slot and underline retain one geometry
        while the notebook changes the displayed label.
        """
        if page_id not in self._contextual:
            return
        labels = tuple(dict.fromkeys(str(label) for label in labels))
        if not labels:
            labels = (str(self._buttons[page_id].cget("text")),)
        if not force and self._label_candidates.get(page_id) == labels:
            return
        self._label_candidates[page_id] = labels
        if self._label_measure is None:
            self._label_measure = ttk.Button(self.tabs, takefocus=False)
        measure = self._label_measure
        width = height = 0
        for style in ("Ribbon.ContextTab.TButton", "Ribbon.ContextTab.Selected.TButton"):
            measure.configure(style=style)
            for label in labels:
                measure.configure(text=label)
                width = max(width, measure.winfo_reqwidth())
                height = max(height, measure.winfo_reqheight())
        size = (width, height + int(self._underlines[page_id].cget("height")))
        if self._reserved_sizes.get(page_id) != size:
            container = self._tab_frames[page_id]
            container.configure(width=size[0], height=size[1])
            container.pack_propagate(False)
            self._reserved_sizes[page_id] = size

    def refresh_label_geometry(self) -> None:
        for page_id, labels in self._label_candidates.items():
            self.reserve_label_space(page_id, labels, force=True)

    def set_selected(self, page_id: str) -> None:
        if self._selected_page_id == page_id:
            return
        self._selected_page_id = page_id
        for key, button in self._buttons.items():
            selected = key == page_id
            contextual = key in self._contextual
            button.configure(
                style=(
                    "Ribbon.ContextTab.Selected.TButton"
                    if contextual and selected
                    else "Ribbon.ContextTab.TButton"
                    if contextual
                    else "Ribbon.Tab.Selected.TButton"
                    if selected
                    else "Ribbon.Tab.TButton"
                )
            )
            self._tab_frames[key].configure(
                style=(
                    "Ribbon.ContextTab.Selected.Container.TFrame"
                    if contextual and selected
                    else "Ribbon.ContextTab.Container.TFrame"
                    if contextual
                    else "Ribbon.Tab.Selected.Container.TFrame"
                    if selected
                    else "Ribbon.Tab.Container.TFrame"
                )
            )
            self._refresh_underline(key)

    def _set_hover(self, page_id: str, entering: bool) -> None:
        previous = self._hovered_page_id
        if entering:
            self._hovered_page_id = page_id
        elif previous == page_id:
            self._hovered_page_id = None
        if previous is not None:
            self._refresh_underline(previous)
        if self._hovered_page_id is not None:
            self._refresh_underline(self._hovered_page_id)

    def _refresh_underline(self, key: str) -> None:
        selected = key == self._selected_page_id
        contextual = key in self._contextual
        hovered = key == self._hovered_page_id and not self._buttons[key].instate(["disabled"])
        self._underlines[key].configure(
            style=(
                "Ribbon.ContextTab.Selected.Underline.TFrame"
                if contextual and selected
                else "Ribbon.Tab.Hover.Underline.TFrame"
                if hovered and not selected
                else "Ribbon.ContextTab.Underline.TFrame"
                if contextual
                else "Ribbon.Tab.Selected.Underline.TFrame"
                if selected
                else "Ribbon.Tab.Underline.TFrame"
            )
        )

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapse_button.configure(text="▾" if collapsed else "▴")

    def tab_button(self, page_id: str) -> ttk.Button:
        return self._buttons[page_id]

    def tab_underline(self, page_id: str) -> ttk.Frame:
        return self._underlines[page_id]


class RibbonBar(ttk.Frame):
    """Reusable compact Ribbon with fixed and contextual page support."""

    BODY_HEIGHT = RibbonPage.DEFAULT_HEIGHT

    def __init__(self, master=None, *, collapsed: bool = False, **kwargs) -> None:
        kwargs.setdefault("style", "Ribbon.Bar.TFrame")
        super().__init__(master, **kwargs)
        self._configure_styles()
        self.body_height = max(
            self.BODY_HEIGHT,
            int(round(self.BODY_HEIGHT * tk_scale_factor(self.winfo_toplevel()))),
        )

        self.pages: dict[str, RibbonPage] = {}
        self._selected_page_id: str | None = None
        self._context_page_id: str | None = None
        self._collapsed = bool(collapsed)
        self._context_notebook = None
        self._context_binding_id = None

        self.tab_strip = RibbonTabStrip(
            self,
            on_select=self.select_page,
            on_toggle_collapsed=self.toggle_collapsed,
        )
        self.tab_strip.pack(side="top", fill="x")

        self.page_host = ttk.Frame(
            self,
            style="Ribbon.PageHost.TFrame",
            height=self.body_height,
        )
        self.page_host.pack_propagate(False)

        # Keep a single-pixel boundary even while collapsed so the Ribbon always
        # reads as a distinct application layer above the document workspace.
        self._bottom_boundary = ttk.Frame(
            self,
            height=1,
            style="Ribbon.BottomBorder.TFrame",
        )
        self._bottom_boundary.pack(side="top", fill="x")
        self._bottom_boundary.pack_propagate(False)

        if not self._collapsed:
            self._show_page_host()
        self.tab_strip.set_collapsed(self._collapsed)

    def _configure_styles(self) -> None:
        configure_ribbon_styles(self)

    def refresh_styles(self) -> None:
        self._configure_styles()
        self.tab_strip.refresh_label_geometry()
        for page in self.pages.values():
            for group in page.groups:
                group.refresh_geometry()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def selected_page_id(self) -> str | None:
        return self._selected_page_id

    @property
    def context_page(self) -> RibbonPage | None:
        if self._context_page_id is None:
            return None
        return self.pages[self._context_page_id]

    def add_page(
        self,
        page_id: str,
        label: str,
        *,
        contextual: bool = False,
    ) -> RibbonPage:
        page_id = str(page_id)
        if page_id in self.pages:
            raise ValueError(f"Ribbon page already exists: {page_id}")
        if contextual and self._context_page_id is not None:
            raise ValueError("RibbonBar supports one contextual page in M1")

        page = RibbonPage(self.page_host, page_id=page_id, height=self.body_height)
        self.pages[page_id] = page
        self.tab_strip.add_tab(page_id, label, contextual=contextual)
        if contextual:
            self._context_page_id = page_id

        if self._selected_page_id is None:
            self.select_page(page_id)
        return page

    def select_page(self, page_id: str) -> None:
        if page_id not in self.pages:
            raise KeyError(f"Unknown Ribbon page: {page_id}")
        if self._selected_page_id == page_id:
            self.tab_strip.set_selected(page_id)
            return

        if self._selected_page_id is not None:
            self.pages[self._selected_page_id].pack_forget()
        self._selected_page_id = page_id
        self.pages[page_id].pack(fill="both", expand=True)
        self.tab_strip.set_selected(page_id)
        self.event_generate("<<RibbonPageChanged>>", when="tail")

    def _show_page_host(self) -> None:
        self.page_host.pack(side="top", fill="x", before=self._bottom_boundary)

    def _hide_page_host(self) -> None:
        self.page_host.pack_forget()

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        if collapsed:
            self._hide_page_host()
        else:
            self._show_page_host()
        self.tab_strip.set_collapsed(collapsed)
        self.event_generate("<<RibbonCollapsedChanged>>", when="tail")

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_context_label(self, label: str) -> None:
        if self._context_page_id is None:
            raise RuntimeError("No contextual Ribbon page has been added")
        self.tab_strip.set_label(self._context_page_id, str(label))

    def select_context_page(self) -> None:
        if self._context_page_id is None:
            raise RuntimeError("No contextual Ribbon page has been added")
        self.select_page(self._context_page_id)

    def bind_context_notebook(
        self,
        notebook: ttk.Notebook,
        *,
        labels: Mapping[str, str] | None = None,
        label_transform: Callable[[str], str] | None = None,
    ) -> None:
        """Mirror the selected outer analysis-tab name to the context tab."""
        if self._context_page_id is None:
            raise RuntimeError("Add a contextual Ribbon page before binding a notebook")

        if self._context_notebook is not None and self._context_binding_id is not None:
            try:
                self._context_notebook.unbind(
                    "<<NotebookTabChanged>>", self._context_binding_id
                )
            except (tk.TclError, TypeError):
                pass

        self._context_notebook = notebook

        def refresh_label(_event=None) -> None:
            try:
                tab_id = notebook.select()
                source = str(notebook.tab(tab_id, "text")) if tab_id else "Analysis"
                candidates = [str(notebook.tab(tab, "text")) for tab in notebook.tabs()]
            except tk.TclError:
                source = "Analysis"
                candidates = [source]
            if labels is not None:
                source = labels.get(source, source)
                candidates = [labels.get(label, label) for label in candidates]
            if label_transform is not None:
                source = label_transform(source)
                candidates = [label_transform(label) for label in candidates]
            self.tab_strip.reserve_label_space(self._context_page_id, candidates)
            self.set_context_label(source)

        self._context_binding_id = notebook.bind(
            "<<NotebookTabChanged>>", refresh_label, add="+"
        )
        refresh_label()
