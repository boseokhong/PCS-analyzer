"""Stable contextual Ribbon group switching driven by a ttk.Notebook."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import tkinter as tk
from tkinter import ttk

from .ribbon_group import RibbonGroup


class ContextRibbonController:
    """Show pre-built Ribbon groups for the active outer analysis tab.

    The controller never creates or destroys command widgets. Context changes
    only pack the already-built groups assigned to the selected notebook tab.
    An optional activation callback runs after a meaningful notebook selection
    change and never during the initial presentation refresh.
    """

    def __init__(
        self,
        notebook: ttk.Notebook,
        contexts: Mapping[str, Sequence[RibbonGroup]],
        *,
        nested_contexts: Mapping[
            str,
            tuple[ttk.Notebook, Mapping[str, Sequence[RibbonGroup]]],
        ] | None = None,
        on_context_changed: Callable[[str], object] | None = None,
        on_context_activated: Callable[[str], object] | None = None,
    ) -> None:
        self.notebook = notebook
        self.contexts = {
            str(label): tuple(groups)
            for label, groups in contexts.items()
        }
        self.nested_contexts = {
            str(outer_label): (
                nested_notebook,
                {
                    str(inner_label): tuple(groups)
                    for inner_label, groups in inner_contexts.items()
                },
            )
            for outer_label, (nested_notebook, inner_contexts) in (
                nested_contexts or {}
            ).items()
        }
        self.groups = tuple(
            dict.fromkeys(
                group
                for groups in (
                    list(self.contexts.values())
                    + [
                        groups
                        for _, inner_contexts in self.nested_contexts.values()
                        for groups in inner_contexts.values()
                    ]
                )
                for group in groups
            )
        )
        self.active_context: str | None = None
        self._active_groups: tuple[RibbonGroup, ...] | None = None
        self.on_context_changed = on_context_changed
        self.on_context_activated = on_context_activated
        self._last_outer_tab_id: str | None = None
        self._last_nested_tab_ids: dict[ttk.Notebook, str] = {}
        self._workspace_bindtag = f"ContextRibbonWorkspace{id(self)}"
        self._workspace_widgets: list[tk.Misc] = []
        self._workspace_activation_ready = False
        self._workspace_activation_after_id = None
        self._binding_id = notebook.bind(
            "<<NotebookTabChanged>>",
            self.refresh,
            add="+",
        )
        self._nested_binding_ids = [
            (
                nested_notebook,
                nested_notebook.bind(
                    "<<NotebookTabChanged>>",
                    self.refresh,
                    add="+",
                ),
            )
            for nested_notebook, _ in self.nested_contexts.values()
        ]
        self._install_workspace_bindings()
        self._workspace_activation_after_id = notebook.after_idle(
            self._enable_workspace_activation
        )
        self.refresh()

    def _install_workspace_bindings(self) -> None:
        """Tag existing analysis-workspace widgets without rebuilding them."""
        try:
            tab_ids = self.notebook.tabs()
        except tk.TclError:
            tab_ids = ()

        def visit(widget) -> None:
            try:
                tags = tuple(widget.bindtags())
                if self._workspace_bindtag not in tags:
                    insert_at = max(0, len(tags) - 1)
                    widget.bindtags(
                        tags[:insert_at]
                        + (self._workspace_bindtag,)
                        + tags[insert_at:]
                    )
                self._workspace_widgets.append(widget)
                children = widget.winfo_children()
            except tk.TclError:
                return
            for child in children:
                visit(child)

        for tab_id in tab_ids:
            try:
                visit(self.notebook.nametowidget(tab_id))
            except (KeyError, tk.TclError):
                pass

        self.notebook.bind_class(
            self._workspace_bindtag,
            "<Button-1>",
            self._on_workspace_interaction,
            add="+",
        )
        self.notebook.bind_class(
            self._workspace_bindtag,
            "<FocusIn>",
            self._on_workspace_interaction,
            add="+",
        )

    def _enable_workspace_activation(self) -> None:
        self._workspace_activation_after_id = None
        self._workspace_activation_ready = True

    def _on_workspace_interaction(self, event) -> None:
        if not self._workspace_activation_ready:
            return
        try:
            selected = self.notebook.select()
            active_workspace = self.notebook.nametowidget(selected)
        except (KeyError, tk.TclError):
            return

        widget = getattr(event, "widget", None)
        while widget is not None:
            if widget is active_workspace:
                # Re-resolve the current nested context and command state, then
                # restore the existing contextual page. This never navigates
                # the analysis notebook or reconstructs Ribbon content.
                self.refresh()
                if self.on_context_activated is not None:
                    self.on_context_activated(self.active_context or "")
                return
            widget = getattr(widget, "master", None)

    def refresh(self, _event=None) -> None:
        try:
            tab_id = self.notebook.select()
            label = str(self.notebook.tab(tab_id, "text")) if tab_id else ""
        except tk.TclError:
            label = ""

        active_groups = self.contexts.get(label, ())
        active_context = label
        outer_changed = (
            _event is not None
            and getattr(_event, "widget", None) is self.notebook
            and tab_id != self._last_outer_tab_id
        )
        nested_changed = False
        nested = self.nested_contexts.get(label)
        if nested is not None:
            nested_notebook, inner_contexts = nested
            inner_tab_id = ""
            try:
                inner_tab_id = nested_notebook.select()
                inner_label = (
                    str(nested_notebook.tab(inner_tab_id, "text"))
                    if inner_tab_id
                    else ""
                )
            except tk.TclError:
                inner_label = ""
            active_groups = inner_contexts.get(inner_label, ())
            active_context = f"{label} / {inner_label}" if inner_label else label
            nested_changed = (
                _event is not None
                and getattr(_event, "widget", None) is nested_notebook
                and inner_tab_id
                != self._last_nested_tab_ids.get(nested_notebook)
            )
            self._last_nested_tab_ids[nested_notebook] = inner_tab_id

        if active_groups != self._active_groups:
            # On first presentation all groups were packed by add_group. Later
            # hide only the previous set, retaining exact ordering (including
            # shared groups) when a genuinely different context is selected.
            for group in self.groups if self._active_groups is None else self._active_groups:
                group.pack_forget()
            for group in active_groups:
                group.pack(side="left", fill="y")
            self._active_groups = active_groups
        pages = {
            getattr(getattr(group, "master", None), "master", None)
            for group in self.groups
        }
        for page in pages:
            schedule = getattr(page, "_schedule_overflow_update", None)
            if callable(schedule):
                schedule()
        self.active_context = active_context
        self._last_outer_tab_id = tab_id
        if self.on_context_changed is not None:
            self.on_context_changed(active_context)
        # Construction performs an initial presentation refresh but is not a
        # navigation action.  Only a real outer/nested selection change asks
        # the Ribbon to reveal its already-built contextual page.
        if (
            (outer_changed or nested_changed)
            and self.on_context_activated is not None
        ):
            self.on_context_activated(active_context)

    def destroy(self) -> None:
        if self._workspace_activation_after_id is not None:
            try:
                self.notebook.after_cancel(self._workspace_activation_after_id)
            except tk.TclError:
                pass
        self._workspace_activation_after_id = None
        for sequence in ("<Button-1>", "<FocusIn>"):
            try:
                self.notebook.unbind_class(self._workspace_bindtag, sequence)
            except tk.TclError:
                pass
        for widget in self._workspace_widgets:
            try:
                widget.bindtags(
                    tuple(
                        tag
                        for tag in widget.bindtags()
                        if tag != self._workspace_bindtag
                    )
                )
            except tk.TclError:
                pass
        self._workspace_widgets = []
        if self._binding_id is not None:
            try:
                self.notebook.unbind(
                    "<<NotebookTabChanged>>",
                    self._binding_id,
                )
            except (tk.TclError, TypeError):
                pass
        self._binding_id = None
        for nested_notebook, binding_id in self._nested_binding_ids:
            try:
                nested_notebook.unbind("<<NotebookTabChanged>>", binding_id)
            except (tk.TclError, TypeError):
                pass
        self._nested_binding_ids = []


__all__ = ["ContextRibbonController"]
