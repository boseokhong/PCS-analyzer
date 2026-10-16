"""Ribbon page container."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.commands import CommandDisabledError
from ui.tooltips import ToolTip

from .ribbon_group import RibbonGroup


class RibbonPage(ttk.Frame):
    """A stable page host whose groups are laid out horizontally."""

    DEFAULT_HEIGHT = 94

    def __init__(self, master=None, *, page_id: str, **kwargs) -> None:
        kwargs.setdefault("style", "Ribbon.Page.TFrame")
        kwargs.setdefault("height", self.DEFAULT_HEIGHT)
        super().__init__(master, **kwargs)
        self.page_id = str(page_id)
        self.pack_propagate(False)

        self.overflow_menu = tk.Menu(self, tearoff=False)
        self.overflow_menu.configure(postcommand=self._rebuild_overflow_menu)
        self.overflow_button = ttk.Menubutton(
            self,
            text="More commands ▾",
            menu=self.overflow_menu,
            style="Ribbon.Overflow.TButton",
            takefocus=True,
        )
        self._overflow_tooltip = ToolTip(
            self.overflow_button,
            "All commands on this Ribbon page. Shown when the window is narrow.",
        )
        self._overflow_active = False
        self._overflow_after_id = None
        self._overflow_submenus = []

        self.group_host = ttk.Frame(self, style="Ribbon.Page.TFrame")
        self.group_host.pack(side="left", fill="both", expand=True, padx=2, pady=0)
        self.groups: list[RibbonGroup] = []
        self.bind("<Configure>", self._schedule_overflow_update, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")

    def add_group(self, caption: str, **kwargs) -> RibbonGroup:
        group = RibbonGroup(self.group_host, caption=caption, **kwargs)
        group.pack(side="left", fill="y")
        self.groups.append(group)
        self._schedule_overflow_update()
        return group

    @property
    def overflow_active(self) -> bool:
        return self._overflow_active

    def _schedule_overflow_update(self, _event=None) -> None:
        if self._overflow_after_id is not None:
            return
        try:
            self._overflow_after_id = self.after_idle(self._update_overflow)
        except tk.TclError:
            self._overflow_after_id = None

    def _update_overflow(self, available_width=None) -> None:
        self._overflow_after_id = None
        try:
            width = int(available_width or self.winfo_width())
            groups = self._visible_groups()
            required = sum(group.winfo_reqwidth() for group in groups) + 10
        except tk.TclError:
            return
        active = width > 1 and required > max(1, width - 8)
        if active == self._overflow_active:
            return
        self._overflow_active = active
        if active:
            self.overflow_button.pack(
                side="right",
                fill="y",
                padx=(2, 3),
                pady=3,
                before=self.group_host,
            )
            self.overflow_button.lift()
        else:
            self.overflow_button.pack_forget()

    def _rebuild_overflow_menu(self) -> None:
        self.overflow_menu.delete(0, "end")
        for submenu in self._overflow_submenus:
            try:
                submenu.destroy()
            except tk.TclError:
                pass
        self._overflow_submenus = []
        added = False
        for group in self._visible_groups():
            commands = []
            supplier = getattr(group, "overflow_commands_supplier", None)
            if callable(supplier):
                commands = list(supplier())
            for button in group.buttons:
                if callable(supplier):
                    break
                command = getattr(button, "command_model", None)
                if command is not None:
                    command.refresh()
                    commands.append(command)
            if not commands:
                continue
            submenu = tk.Menu(self.overflow_menu, tearoff=False)
            self._overflow_submenus.append(submenu)
            for command in commands:
                command.refresh()
                label = (
                    f"✓ {command.label}"
                    if command.is_toggle and command.checked
                    else command.label
                )
                submenu.add_command(
                    label=label,
                    state="normal" if command.enabled else "disabled",
                    command=lambda current=command: self._invoke_overflow(current),
                )
            self.overflow_menu.add_cascade(label=group.caption, menu=submenu)
            added = True
        if not added:
            self.overflow_menu.add_command(label="No commands", state="disabled")

    def _visible_groups(self):
        return tuple(group for group in self.groups if group.winfo_manager())

    @staticmethod
    def _invoke_overflow(command) -> None:
        try:
            command.invoke()
        except CommandDisabledError:
            command.refresh(force=True)

    def _on_destroy(self, event) -> None:
        if event.widget is not self or self._overflow_after_id is None:
            return
        try:
            self.after_cancel(self._overflow_after_id)
        except (tk.TclError, ValueError):
            pass
        self._overflow_after_id = None
