"""Button presenter used inside Ribbon groups."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.commands import Command, CommandDisabledError
from ui.tooltips import ToolTip, command_tooltip_text


class RibbonButton(ttk.Button):
    """A compact Ribbon command button with a guarded Tk image reference.

    M1 deliberately keeps this widget command-agnostic. Later milestones can
    bind it to a shared command registry without changing Ribbon layout code.
    """

    def __init__(
        self,
        master=None,
        *,
        text: str = "",
        command=None,
        image=None,
        command_model: Command | None = None,
        icon_resolver=None,
        compound=tk.TOP,
        style: str = "Ribbon.Small.TButton",
        **kwargs,
    ) -> None:
        self._ribbon_image = image
        self._command_model: Command | None = None
        self._command_unsubscribe = None
        self._command_description = ""
        self._icon_resolver = icon_resolver
        self._presentation_role = "secondary"
        options = dict(
            text=text,
            command=command,
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
        """Description retained for the shared tooltip presenter planned later."""
        return self._command_description

    @property
    def presentation_role(self) -> str:
        return self._presentation_role

    def bind_command(
        self,
        command: Command,
        *,
        image=None,
        icon_resolver=None,
    ) -> None:
        """Present one shared command and observe its enabled state."""
        if not isinstance(command, Command):
            raise TypeError("command must be a Command instance")
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
                        command.icon, role=command.presentation
                    )
                except TypeError:
                    resolved_image = self._icon_resolver(command.icon)
            elif not isinstance(command.icon, str):
                resolved_image = command.icon
        self.set_image(resolved_image)
        self._apply_presentation(command.presentation)
        self._tooltip.update_text(command_tooltip_text(command))
        self.configure(command=self._invoke_bound_command)
        self._command_unsubscribe = command.subscribe(self._apply_command_state)

    def unbind_command(self) -> None:
        if self._command_unsubscribe is not None:
            self._command_unsubscribe()
        self._command_unsubscribe = None
        self._command_model = None
        self._command_description = ""
        if hasattr(self, "_tooltip"):
            self._tooltip.update_text("")

    def _apply_presentation(self, role: str) -> None:
        self._presentation_role = str(role or "secondary")
        if self._presentation_role == "primary":
            style, compound = "Ribbon.Primary.TButton", tk.TOP
        elif self._presentation_role == "danger":
            style, compound = "Ribbon.Danger.TButton", tk.LEFT
        else:
            style, compound = "Ribbon.Small.TButton", tk.LEFT
        self.configure(style=style, compound=compound)


    def _primary_label(self, text: str) -> str:
        """Normalize Primary labels without forcing a second text line.

        Primary buttons keep their fixed width and the style-owned wraplength
        may use a second line when needed.  Short labels remain one line, so
        the common button height is preserved without inserting an artificial
        blank line into the content block.
        """
        return " ".join(str(text or "").split())

    def _display_label(self, text: str) -> str:
        if self._presentation_role == "primary":
            return self._primary_label(text)
        return str(text)

    def _apply_command_state(self, command: Command) -> None:
        try:
            self.configure(
                text=self._display_label(command.ribbon_label),
                state="normal" if command.enabled else "disabled",
            )
            self._tooltip.update_text(command_tooltip_text(command))
        except tk.TclError:
            self.unbind_command()

    def _invoke_bound_command(self):
        if self._command_model is None:
            return None
        try:
            return self._command_model.invoke()
        except CommandDisabledError:
            self._command_model.refresh(force=True)
            return None

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self.unbind_command()

    def set_image(self, image) -> None:
        """Update the displayed image while retaining a Python reference."""
        self._ribbon_image = image
        self.configure(image=image if image not in (None, "") else "")
