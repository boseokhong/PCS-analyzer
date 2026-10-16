"""Centralized PNG icon loading/caching for the PCS Analyzer UI.

The manager deliberately owns PhotoImage references so Tk does not garbage-
collect images that are still used by ttk widgets.  Callers refer to semantic
icon IDs and request a display size; they do not resolve file paths directly.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk


TAB_ICON_SIZE = 20
TOOLBAR_ICON_SIZE = 20
BUTTON_ICON_SIZE = 20
LARGE_ICON_SIZE = 24
SUPPORTED_ICON_SIZES = (16, 20, 24, 32, 64)


def _resource_candidates() -> list[Path]:
    """Return possible application roots for source and frozen builds."""
    candidates: list[Path] = []

    # Normal source checkout / onedir bundle that preserves package layout.
    try:
        candidates.append(Path(__file__).resolve().parents[1])
    except Exception:
        pass

    # PyInstaller one-file temporary bundle.
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root))

    # Executable-adjacent resources used by the existing resource_path policy.
    if getattr(sys, "frozen", False):
        try:
            exe_dir = Path(sys.executable).resolve().parent
            candidates.extend((exe_dir, exe_dir / "lib"))
        except Exception:
            pass

    # Keep order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


class IconManager:
    """Load and cache semantic UI icons.

    Runtime Pillow is intentionally not required.  Exact pre-sized PNG variants
    are preferred (img/icons/<size>/<name>.png).  The original source PNG is a
    fallback for development or incomplete asset sets.
    """

    def __init__(self, root: tk.Misc, icon_dir: str | Path | None = None) -> None:
        self.root = root
        self._cache: dict[tuple[str, int], tk.PhotoImage] = {}
        self._base_dirs: list[Path] = []

        if icon_dir is not None:
            self._base_dirs.append(Path(icon_dir))
        else:
            self._base_dirs.extend(root_dir / "img" / "icons" for root_dir in _resource_candidates())

        # Keep the manager itself reachable from the Tk root as an additional
        # lifetime guard for PhotoImage instances.
        try:
            setattr(root, "_icon_manager", self)
        except Exception:
            pass

    @staticmethod
    def _normalize_size(size: int | str | None) -> int:
        try:
            value = int(size or BUTTON_ICON_SIZE)
        except Exception:
            value = BUTTON_ICON_SIZE
        return max(min(SUPPORTED_ICON_SIZES), value)

    def _find_icon(self, name: str, size: int) -> Path | None:
        filename = f"{name}.png"
        for base in self._base_dirs:
            sized = base / str(size) / filename
            if sized.exists():
                return sized
            original = base / filename
            if original.exists():
                return original
        return None

    def available_sizes(self, name: str) -> tuple[int, ...]:
        """Return exact packaged sizes available for one semantic icon key."""
        filename = f"{str(name or '').strip()}.png"
        if filename == ".png":
            return ()
        available = {
            size
            for base in self._base_dirs
            for size in SUPPORTED_ICON_SIZES
            if (base / str(size) / filename).exists()
        }
        return tuple(sorted(available))

    def get(self, name: str, size: int = BUTTON_ICON_SIZE):
        """Return a cached PhotoImage, or an empty image name when unavailable."""
        icon_name = str(name or "").strip()
        if not icon_name:
            return ""

        px = self._normalize_size(size)
        # Small history assets are opt-in. Preserve the former 20px minimum
        # for existing icons/plugins without an exact small packaged variant.
        if px < TAB_ICON_SIZE and px not in self.available_sizes(icon_name):
            px = TAB_ICON_SIZE
        key = (icon_name, px)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        path = self._find_icon(icon_name, px)
        if path is None:
            return ""

        try:
            image = tk.PhotoImage(master=self.root, file=str(path))
        except Exception:
            return ""

        # If only a large source PNG exists, use Tk's integer subsampling as a
        # dependency-free fallback.  Packaged assets normally include exact
        # 20/24/28 px variants so this path should be rare.
        try:
            w, h = int(image.width()), int(image.height())
            if max(w, h) > px * 2:
                factor = max(1, int(round(max(w, h) / float(px))))
                image = image.subsample(factor, factor)
        except Exception:
            pass

        self._cache[key] = image
        return image

    def clear(self) -> None:
        self._cache.clear()


__all__ = [
    "IconManager",
    "TAB_ICON_SIZE",
    "TOOLBAR_ICON_SIZE",
    "BUTTON_ICON_SIZE",
    "LARGE_ICON_SIZE",
    "SUPPORTED_ICON_SIZES",
]
