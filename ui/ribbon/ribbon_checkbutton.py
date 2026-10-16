"""Compact native-checkbox presenter for Ribbon option groups."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ui.commands import Command
from ui.tooltips import ToolTip, command_tooltip_text


class RibbonCheckbutton(ttk.Checkbutton):
    """A native checkbox with an existing variable or command-owned state.

    Supplied variables retain their existing interaction contract. Without a
    variable, mirror the command's checked state and invoke its callback;
    application data remains authoritative across document/settings changes.
    """

    def __init__(
        self,
        master=None,
        *,
        command_model: Command,
        variable: tk.Variable | None = None,
        on_change: Callable[[], object] | None = None,
        style: str = "Ribbon.Checkbutton.TCheckbutton",
        **kwargs,
    ) -> None:
        if not isinstance(command_model, Command):
            raise TypeError("command_model must be a Command instance")
        if not command_model.is_toggle:
            raise ValueError("RibbonCheckbutton requires a toggle command")
        self._mirrors_command = variable is None
        if variable is None:
            variable = tk.BooleanVar(master=master, value=command_model.checked)
        if not isinstance(variable, tk.Variable):
            raise TypeError("variable must be an existing Tk variable")

        self._command_model = command_model
        self._state_variable = variable
        self._on_change = on_change
        self._command_unsubscribe = None
        super().__init__(
            master,
            variable=variable,
            command=self._changed,
            style=style,
            takefocus=True,
            **kwargs,
        )
        self._tooltip = ToolTip(self, command_tooltip_text(command_model))
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._command_unsubscribe = command_model.subscribe(
            self._apply_command_state
        )

    @property
    def command_model(self) -> Command:
        return self._command_model

    @property
    def command_description(self) -> str:
        return self._command_model.description

    @property
    def presentation_role(self) -> str:
        return "secondary"

    @property
    def state_variable(self) -> tk.Variable:
        return self._state_variable

    def _apply_command_state(self, command: Command) -> None:
        try:
            if self._mirrors_command:
                self._state_variable.set(bool(command.checked))
            self.configure(
                text=command.label,
                state="normal" if command.enabled else "disabled",
            )
            self._tooltip.update_text(command_tooltip_text(command))
        except tk.TclError:
            self._unsubscribe()

    def _changed(self):
        if self._on_change is not None:
            result = self._on_change()
        elif self._mirrors_command:
            result = self._command_model.invoke()
        else:
            result = None
        self._command_model.refresh(force=True)
        return result

    def _unsubscribe(self) -> None:
        if self._command_unsubscribe is not None:
            self._command_unsubscribe()
            self._command_unsubscribe = None

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._unsubscribe()


__all__ = ["RibbonCheckbutton"]
