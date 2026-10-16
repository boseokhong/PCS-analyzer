"""Shared UI command model and registry.

Commands describe presentation and invoke callbacks owned elsewhere. They do
not own PCS Analyzer scientific state. Optional enabled and checked suppliers
let existing state owners remain authoritative while presenters observe a
small, dependency-free model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Union


BoolSource = Union[bool, Callable[[], bool]]
CommandObserver = Callable[["Command"], None]


class DuplicateCommandError(ValueError):
    """Raised when a registry already contains the requested command ID."""


class UnknownCommandError(KeyError):
    """Raised when a command ID is not registered."""


class CommandDisabledError(RuntimeError):
    """Raised when code attempts to invoke a disabled command."""


class NonToggleCommandError(ValueError):
    """Raised when checked state is assigned to a non-toggle command."""


class Command:
    """One shareable UI action with observable presentation state.

    ``enabled`` and ``checked`` may be plain booleans or zero-argument
    suppliers. Suppliers are evaluated by :meth:`refresh`, including directly
    before invocation. They should read existing application state rather than
    duplicate it in the command.
    """

    def __init__(
        self,
        command_id: str,
        label: str,
        callback: Callable[..., Any],
        *,
        icon: Any = None,
        description: str = "",
        shortcut: str | None = None,
        enabled: BoolSource = True,
        checked: BoolSource | None = None,
        category: str | None = None,
        group: str | None = None,
        presentation: str = "secondary",
        ribbon_label: str | None = None,
    ) -> None:
        normalized_id = str(command_id or "").strip()
        if not normalized_id:
            raise ValueError("command_id must be a non-empty string")
        if not callable(callback):
            raise TypeError("callback must be callable")

        self.command_id = normalized_id
        self.label = str(label)
        self.callback = callback
        self.icon = icon
        self.description = str(description or "")
        self.shortcut = shortcut
        self.category = category
        self.group = group
        self.ribbon_label = str(ribbon_label or label)
        role = str(presentation or "secondary").strip().lower()
        if role not in {"primary", "secondary", "danger"}:
            raise ValueError(
                "presentation must be 'primary', 'secondary', or 'danger'"
            )
        self.presentation = role

        self._enabled_source = self._validate_bool_source("enabled", enabled)
        self._checked_source = (
            None
            if checked is None
            else self._validate_bool_source("checked", checked)
        )
        self._enabled = self._read_bool(self._enabled_source)
        self._checked = (
            None
            if self._checked_source is None
            else self._read_bool(self._checked_source)
        )
        self._observers: dict[int, CommandObserver] = {}
        self._next_observer_token = 1

    @staticmethod
    def _validate_bool_source(name: str, source: BoolSource) -> BoolSource:
        if callable(source) or isinstance(source, bool):
            return source
        raise TypeError(f"{name} must be a bool or zero-argument callable")

    @staticmethod
    def _read_bool(source: BoolSource) -> bool:
        return bool(source() if callable(source) else source)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_toggle(self) -> bool:
        return self._checked_source is not None

    @property
    def checked(self) -> bool | None:
        return self._checked

    def invoke(self, *args, **kwargs):
        """Invoke the existing callback if the latest state allows it."""
        self.refresh()
        if not self._enabled:
            raise CommandDisabledError(
                f"Command is disabled: {self.command_id}"
            )
        return self.callback(*args, **kwargs)

    def set_enabled(self, enabled: bool) -> None:
        """Set simple presentation enablement and notify all presenters.

        Commands backed by an application-state supplier should update that
        owner and call :meth:`refresh` instead.
        """
        self._enabled_source = bool(enabled)
        self.refresh()

    def set_checked(self, checked: bool) -> None:
        """Set simple toggle presentation state and notify all presenters."""
        if self._checked_source is None:
            raise NonToggleCommandError(
                f"Command is not a toggle: {self.command_id}"
            )
        self._checked_source = bool(checked)
        self.refresh()

    def refresh(self, *, force: bool = False) -> bool:
        """Re-read state suppliers and notify observers when state changes.

        Returns ``True`` when cached presentation state changed.
        """
        enabled = self._read_bool(self._enabled_source)
        checked = (
            None
            if self._checked_source is None
            else self._read_bool(self._checked_source)
        )
        changed = enabled != self._enabled or checked != self._checked
        self._enabled = enabled
        self._checked = checked
        if changed or force:
            self._notify_observers()
        return changed

    def subscribe(
        self,
        observer: CommandObserver,
        *,
        notify_immediately: bool = True,
    ) -> Callable[[], None]:
        """Observe presentation state; return an idempotent unsubscribe hook."""
        if not callable(observer):
            raise TypeError("observer must be callable")
        token = self._next_observer_token
        self._next_observer_token += 1
        self._observers[token] = observer
        if notify_immediately:
            observer(self)

        def unsubscribe() -> None:
            self._observers.pop(token, None)

        return unsubscribe

    def _notify_observers(self) -> None:
        # Snapshot permits an observer to unsubscribe itself safely.
        for observer in tuple(self._observers.values()):
            observer(self)


class CommandRegistry:
    """Stable-ID collection and dispatch point for shared UI commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> Command:
        if not isinstance(command, Command):
            raise TypeError("command must be a Command instance")
        if command.command_id in self._commands:
            raise DuplicateCommandError(command.command_id)
        self._commands[command.command_id] = command
        return command

    def create(
        self,
        command_id: str,
        label: str,
        callback: Callable[..., Any],
        **metadata,
    ) -> Command:
        """Construct, register, and return a command."""
        return self.register(Command(command_id, label, callback, **metadata))

    def get(self, command_id: str) -> Command:
        try:
            return self._commands[str(command_id)]
        except KeyError as exc:
            raise UnknownCommandError(command_id) from exc

    def unregister(self, command_id: str) -> Command:
        try:
            return self._commands.pop(str(command_id))
        except KeyError as exc:
            raise UnknownCommandError(command_id) from exc

    def invoke(self, command_id: str, *args, **kwargs):
        return self.get(command_id).invoke(*args, **kwargs)

    def refresh(self, command_id: str | None = None, *, force: bool = False) -> None:
        commands = (
            (self.get(command_id),)
            if command_id is not None
            else tuple(self._commands.values())
        )
        for command in commands:
            command.refresh(force=force)

    def subscribe(
        self,
        command_id: str,
        observer: CommandObserver,
        *,
        notify_immediately: bool = True,
    ) -> Callable[[], None]:
        return self.get(command_id).subscribe(
            observer,
            notify_immediately=notify_immediately,
        )

    def __contains__(self, command_id: object) -> bool:
        return str(command_id) in self._commands

    def __len__(self) -> int:
        return len(self._commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self._commands.values())

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._commands)


__all__ = [
    "Command",
    "CommandRegistry",
    "CommandDisabledError",
    "DuplicateCommandError",
    "NonToggleCommandError",
    "UnknownCommandError",
]
