"""Launcher registry shared by built-ins and existing Python plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolLauncher:
    id: str
    label: str
    category: str
    launch: Callable[[], object]
    source: str = "builtin"
    enabled: Callable[[], bool] | None = None
    order: int = 0
    tooltip: str = ""

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("tool id must not be blank")
        if not str(self.label).strip():
            raise ValueError("tool label must not be blank")
        if not callable(self.launch):
            raise TypeError("tool launch callback must be callable")

    def is_enabled(self) -> bool:
        return True if self.enabled is None else bool(self.enabled())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolLauncher] = {}

    def register(self, tool: ToolLauncher, *, replace: bool = True) -> ToolLauncher:
        if not isinstance(tool, ToolLauncher):
            raise TypeError("tool must be a ToolLauncher")
        if not replace and tool.id in self._tools:
            raise ValueError(f"tool already registered: {tool.id}")
        self._tools[tool.id] = tool
        return tool

    def unregister(self, tool_id: str) -> None:
        self._tools.pop(str(tool_id), None)

    def unregister_source(self, source: str) -> None:
        for tool_id in tuple(self._tools):
            if self._tools[tool_id].source == str(source):
                self._tools.pop(tool_id, None)

    def unregister_plugins(self) -> None:
        for tool_id in tuple(self._tools):
            if self._tools[tool_id].source != "builtin":
                self._tools.pop(tool_id, None)

    def get(self, tool_id: str) -> ToolLauncher | None:
        return self._tools.get(str(tool_id))

    def available(self) -> tuple[ToolLauncher, ...]:
        return tuple(
            sorted(
                self._tools.values(),
                key=lambda item: (item.category.lower(), item.order, item.label.lower()),
            )
        )

    def pinned(self, tool_ids) -> tuple[ToolLauncher, ...]:
        output = []
        for tool_id in tool_ids:
            tool = self.get(str(tool_id))
            if tool is not None:
                output.append(tool)
        return tuple(output)


def ensure_tool_registry(state: dict) -> ToolRegistry:
    registry = state.get("tool_registry")
    if registry is None:
        registry = ToolRegistry()
        state["tool_registry"] = registry
    elif not isinstance(registry, ToolRegistry):
        raise TypeError("state['tool_registry'] must be a ToolRegistry")
    return registry


__all__ = ["ToolLauncher", "ToolRegistry", "ensure_tool_registry"]
