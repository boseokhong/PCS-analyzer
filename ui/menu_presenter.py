"""Small Tk menu presenter for a shared :class:`ui.commands.Command`."""

from __future__ import annotations

import tkinter as tk

from ui.commands import Command, CommandDisabledError


class MenuCommandPresenter:
    """Keep one native menu entry synchronized with a shared command."""

    def __init__(self, menu: tk.Menu, command: Command) -> None:
        if not isinstance(command, Command):
            raise TypeError("command must be a Command instance")
        self.menu = menu
        self.command = command
        menu.add_command(command=self._invoke)
        self.index = int(menu.index("end"))
        self._unsubscribe = command.subscribe(self._apply_state)

    def _apply_state(self, command: Command) -> None:
        try:
            self.menu.entryconfigure(
                self.index,
                label=command.label,
                state="normal" if command.enabled else "disabled",
                accelerator=str(command.shortcut or ""),
            )
        except tk.TclError:
            self.destroy()

    def _invoke(self):
        try:
            return self.command.invoke()
        except CommandDisabledError:
            self.command.refresh(force=True)
            return None

    def destroy(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None


__all__ = ["MenuCommandPresenter"]
