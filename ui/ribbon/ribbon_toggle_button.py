"""Toggle presenter for observable shared commands."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.commands import Command, CommandDisabledError
from ui.tooltips import ToolTip, command_tooltip_text


class RibbonToggleButton(ttk.Checkbutton):
    """A small Ribbon toggle whose state is supplied by a shared command.

    Invoking the widget never makes it the scientific state owner. The command
    callback updates the existing owner; ``Command.refresh()`` then mirrors the
    authoritative checked supplier back to every presenter.
    """

    def __init__(
        self,
        master=None,
        *,
        command_model: Command | None = None,
        image=None,
        icon_resolver=None,
        compound=tk.LEFT,
        style: str = "Ribbon.Toggle.TCheckbutton",
        **kwargs,
    ) -> None:
        self.variable = tk.BooleanVar(master=master, value=False)
        self._ribbon_image = image
        self._command_model: Command | None = None
        self._command_unsubscribe = None
        self._command_description = ""
        self._icon_resolver = icon_resolver
        options = dict(
            variable=self.variable,
            command=self._invoke_bound_command,
            compound=compound,
            style=style,
            takefocus=True,
        )
        if image not in (None, ""):
            options["image"] = image
        options.update(kwargs)
        super().__init__(master, **options)
        self._tooltip = ToolTip(self, "")
        self.bind("<Destroy>", self._on_destroy, add="+")
        if command_model is not None:
            self.bind_command(
                command_model,
                image=image,
                icon_resolver=icon_resolver,
            )

    @property
    def command_model(self) -> Command | None:
        return self._command_model

    @property
    def command_description(self) -> str:
        return self._command_description

    @property
    def presentation_role(self) -> str:
        return "toggle"

    def bind_command(
        self,
        command: Command,
        *,
        image=None,
        icon_resolver=None,
    ) -> None:
        if not isinstance(command, Command):
            raise TypeError("command must be a Command instance")
        if not command.is_toggle:
            raise ValueError("RibbonToggleButton requires a toggle command")
        self.unbind_command()
        self._command_model = command
        self._command_description = command.description
        if icon_resolver is not None:
            self._icon_resolver = icon_resolver

        resolved_image = image
        if resolved_image in (None, "") and command.icon is not None:
            if callable(self._icon_resolver):
                try:
                    resolved_image = self._icon_resolver(
                        command.icon, role="toggle"
                    )
                except TypeError:
                    resolved_image = self._icon_resolver(command.icon)
            elif not isinstance(command.icon, str):
                resolved_image = command.icon
        self.set_image(resolved_image)
        self._tooltip.update_text(command_tooltip_text(command))
        self._command_unsubscribe = command.subscribe(self._apply_command_state)

    def unbind_command(self) -> None:
        if self._command_unsubscribe is not None:
            self._command_unsubscribe()
        self._command_unsubscribe = None
        self._command_model = None
        self._command_description = ""
        if hasattr(self, "_tooltip"):
            self._tooltip.update_text("")

    def _apply_command_state(self, command: Command) -> None:
        try:
            self.variable.set(bool(command.checked))
            label = command.ribbon_label
            self.configure(
                text=(f"✓ {label}" if command.checked else label),
                state="normal" if command.enabled else "disabled",
            )
            self._tooltip.update_text(command_tooltip_text(command))
        except tk.TclError:
            self.unbind_command()

    def _invoke_bound_command(self):
        if self._command_model is None:
            return None
        try:
            result = self._command_model.invoke()
        except CommandDisabledError:
            result = None
        finally:
            if self._command_model is not None:
                # The callback may have changed an external checked supplier.
                self._command_model.refresh(force=True)
        return result

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self.unbind_command()

    def set_image(self, image) -> None:
        self._ribbon_image = image
        self.configure(image=image if image not in (None, "") else "")
