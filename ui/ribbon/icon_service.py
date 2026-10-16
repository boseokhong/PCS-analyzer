"""DPI-aware packaged icon selection for Ribbon presenters."""

from __future__ import annotations

from ui.icons import SUPPORTED_ICON_SIZES


TK_SCALING_AT_100_PERCENT = 96.0 / 72.0


def tk_scale_factor(root) -> float:
    """Return Tk scaling normalized to 96-DPI Windows logical pixels."""
    try:
        tk_scaling = float(root.tk.call("tk", "scaling"))
        return max(0.5, min(4.0, tk_scaling / TK_SCALING_AT_100_PERCENT))
    except Exception:
        return 1.0


def nearest_packaged_icon_size(
    logical_size: int,
    scale_factor: float,
    supported=SUPPORTED_ICON_SIZES,
) -> int:
    """Choose the nearest packaged PNG size, preferring larger ties."""
    target = max(1.0, float(logical_size) * max(0.5, float(scale_factor)))
    sizes = tuple(sorted(int(size) for size in supported))
    if not sizes:
        return max(1, int(round(target)))
    return min(sizes, key=lambda size: (abs(size - target), -size))


class RibbonIconService:
    """Resolve semantic icon keys through the application's IconManager."""

    LOGICAL_SIZES = {
        "primary": 32,
        "large": 32,
        "secondary": 20,
        "danger": 20,
        "toggle": 20,
    }

    def __init__(self, icon_manager, root, *, scale_factor=None):
        self.icon_manager = icon_manager
        self.root = root
        self.scale_factor = (
            self._read_scale_factor() if scale_factor is None else float(scale_factor)
        )
        self._references = {}

    def _read_scale_factor(self) -> float:
        return tk_scale_factor(self.root)

    def size_for(self, role="secondary", *, name=None) -> int:
        logical = self.LOGICAL_SIZES.get(str(role), 20)
        supported = SUPPORTED_ICON_SIZES
        available_sizes = getattr(self.icon_manager, "available_sizes", None)
        if name and callable(available_sizes):
            available = tuple(available_sizes(name))
            if available:
                supported = available
        return nearest_packaged_icon_size(
            logical,
            self.scale_factor,
            supported=supported,
        )

    def resolve(self, name, *, role="secondary"):
        key = str(name or "").strip()
        if not key or self.icon_manager is None:
            return ""
        size = self.size_for(role, name=key)
        image = self.icon_manager.get(key, size=size)
        if image not in (None, ""):
            self._references[(key, size)] = image
        return image


__all__ = [
    "RibbonIconService",
    "TK_SCALING_AT_100_PERCENT",
    "nearest_packaged_icon_size",
    "tk_scale_factor",
]
