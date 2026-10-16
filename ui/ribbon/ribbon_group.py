"""Captioned group container for Ribbon page content."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .icon_service import tk_scale_factor
from .ribbon_button import RibbonButton
from .ribbon_checkbutton import RibbonCheckbutton
from .ribbon_toggle_button import RibbonToggleButton


class RibbonGroup(ttk.Frame):
    """A compact captioned command zone on a continuous Ribbon surface.

    Layout variants never recreate command widgets:

    ``auto`` / ``primary_stack``
        Large primary commands span all three command rows; secondary commands
        form compact three-row stacks to their right.
    ``primary_row``
        Commands are arranged as full-height cells in one horizontal row.
    ``compact_stack`` / ``tool_grid``
        Commands form compact three-row stacks without large geometry.
    """

    LAYOUT_VARIANTS = {
        "auto",
        "primary_stack",
        "primary_row",
        "compact_stack",
        "tool_grid",
    }
    COMPACT_ROWS = 3
    PRIMARY_LOGICAL_WIDTH = 76

    def __init__(
        self,
        master=None,
        *,
        caption: str,
        layout: str = "auto",
        show_separator: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault("style", "Ribbon.Group.TFrame")
        self._default_padding = "padding" not in kwargs
        scale = tk_scale_factor(master)
        kwargs.setdefault("padding", (round(10 * scale), 0, round(10 * scale), 0))
        super().__init__(master, **kwargs)

        normalized_layout = str(layout or "auto").strip().lower()
        if normalized_layout not in self.LAYOUT_VARIANTS:
            raise ValueError(f"Unknown RibbonGroup layout: {layout}")
        self.layout_variant = normalized_layout
        self._primary_cell_width = max(
            self.PRIMARY_LOGICAL_WIDTH,
            int(round(self.PRIMARY_LOGICAL_WIDTH * tk_scale_factor(self.winfo_toplevel()))),
        )

        if show_separator:
            self.separator = ttk.Separator(
                self, orient="vertical", style="Ribbon.Separator.TSeparator"
            )
            self.separator.pack(side="right", fill="y", padx=(3, 0),
                                pady=(round(12 * scale), round(24 * scale)))
        else:
            self.separator = None

        self.content = ttk.Frame(self, style="Ribbon.Group.TFrame")

        self.caption_band = ttk.Frame(self, style="Ribbon.Group.TFrame")
        self.caption_band.pack(side="bottom", fill="x")
        self.caption_separator = None

        self.caption_var = tk.StringVar(master=self, value=str(caption))
        self.caption_label = ttk.Label(
            self.caption_band,
            textvariable=self.caption_var,
            anchor="center",
            style="Ribbon.GroupCaption.TLabel",
        )
        self.caption_label.pack(fill="x")
        self.content.pack(side="top", fill="both", expand=True, padx=(0, 1), pady=0)
        self.buttons: list[ttk.Widget] = []
        self._configured_columns = 0

    def refresh_geometry(self) -> None:
        """Rescale only presentation spacing; retain widgets and command layout."""
        scale = tk_scale_factor(self.winfo_toplevel())
        if self._default_padding:
            self.configure(padding=(round(10 * scale), 0, round(10 * scale), 0))
        if self.separator is not None:
            self.separator.pack_configure(pady=(round(12 * scale), round(24 * scale)))

    @property
    def caption(self) -> str:
        return self.caption_var.get()

    @caption.setter
    def caption(self, value: str) -> None:
        self.caption_var.set(str(value))

    def add_button(self, **kwargs) -> RibbonButton:
        button = RibbonButton(self.content, **kwargs)
        self.buttons.append(button)
        self._layout_buttons()
        return button

    def add_toggle_button(self, **kwargs) -> RibbonToggleButton:
        button = RibbonToggleButton(self.content, **kwargs)
        self.buttons.append(button)
        self._layout_buttons()
        return button

    def add_checkbutton(self, **kwargs) -> RibbonCheckbutton:
        button = RibbonCheckbutton(self.content, **kwargs)
        self.buttons.append(button)
        self._layout_buttons()
        return button

    def _reset_grid(self) -> None:
        for column in range(self._configured_columns):
            self.content.columnconfigure(column, weight=0, minsize=0)
        for row in range(self.COMPACT_ROWS):
            self.content.rowconfigure(
                row,
                weight=1,
                minsize=0,
                uniform="ribbon_command_row",
            )

    def _layout_buttons(self) -> None:
        """Apply the selected three-row grammar without rebuilding widgets."""
        for button in self.buttons:
            manager = button.winfo_manager()
            if manager == "pack":
                button.pack_forget()
            elif manager == "grid":
                button.grid_forget()

        self._reset_grid()

        primary = [
            button for button in self.buttons
            if getattr(button, "presentation_role", "secondary") == "primary"
        ]
        compact = [button for button in self.buttons if button not in primary]
        variant = self.layout_variant
        if variant == "auto":
            variant = "primary_stack" if primary else "compact_stack"

        column = 0
        if variant == "primary_stack":
            for button in primary:
                self.content.columnconfigure(
                    column,
                    weight=0,
                    minsize=self._primary_cell_width,
                )
                button.grid(
                    row=0,
                    column=column,
                    rowspan=self.COMPACT_ROWS,
                    sticky="nsew",
                    padx=(1, 2),
                    pady=(0, 1),
                )
                column += 1
            compact_columns = self._grid_compact(compact, start_column=column)
            column += compact_columns
        elif variant == "primary_row":
            for button in self.buttons:
                if getattr(button, "presentation_role", "secondary") == "primary":
                    self.content.columnconfigure(
                        column,
                        weight=0,
                        minsize=self._primary_cell_width,
                    )
                button.grid(
                    row=0,
                    column=column,
                    rowspan=self.COMPACT_ROWS,
                    sticky="nsew",
                    padx=1,
                    pady=(0, 1),
                )
                column += 1
        else:
            column = self._grid_compact(self.buttons, start_column=0)

        self._configured_columns = max(1, column)

    def _grid_compact(self, buttons, *, start_column: int) -> int:
        if not buttons:
            return 0
        for index, button in enumerate(buttons):
            if isinstance(button, RibbonCheckbutton) and str(button.cget("style")) in (
                "Ribbon.Checkbutton.TCheckbutton", "Aligned.Ribbon.Checkbutton.TCheckbutton"
            ):
                first = index // self.COMPACT_ROWS * self.COMPACT_ROWS
                mixed = any(isinstance(peer, RibbonButton) and
                            peer._ribbon_image not in (None, "")
                            for peer in buttons[first:first + self.COMPACT_ROWS])
                desired = ("Aligned.Ribbon.Checkbutton.TCheckbutton" if mixed
                           else "Ribbon.Checkbutton.TCheckbutton")
                if str(button.cget("style")) != desired:
                    button.configure(style=desired)
            button.grid(
                row=index % self.COMPACT_ROWS,
                column=start_column + index // self.COMPACT_ROWS,
                sticky="ew",
                padx=1,
                pady=(0, 1),
            )
        return (len(buttons) + self.COMPACT_ROWS - 1) // self.COMPACT_ROWS
