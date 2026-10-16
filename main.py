'''
===============================================================================
                                  PCS Analyzer
===============================================================================

A desktop tool for paramagnetic NMR and pseudocontact-shift analysis,
with molecular visualization and fitting workflows.

-------------------------------------------------------------------------------
2026.10. Boseok Hong
Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR
Author email: <bshong66@gmail.com>
-------------------------------------------------------------------------------

GitHub repository: <https://github.com/boseokhong/PCS-analyzer>
Documentation:     <https://boseokhong.github.io/PCS-analyzer/>

Dependencies:      requirements.txt
Version history:   docs/reference/changelog.md

===============================================================================
'''

from pathlib import Path
import sys
from multiprocessing import freeze_support

from app_version import APP_NAME, APP_VERSION


def resource_path(relative_path: str) -> Path:
    rel = Path(relative_path)

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent

        candidates = [
            exe_dir / rel,
            exe_dir / "lib" / rel,
        ]

        for path in candidates:
            if path.exists():
                return path

        return candidates[0]

    return Path(__file__).resolve().parent / rel


def apply_window_icon(root):
    icon_path = resource_path("img/logoheader.ico")

    if not icon_path.exists():
        return

    try:
        root.iconbitmap(str(icon_path))
    except Exception:
        pass


def main(*, molecule_viewer_host_probe=False):
    from ui.components import build_app, wire

    state = build_app()

    state["app_name"] = APP_NAME
    state["app_version"] = APP_VERSION

    try:
        state["root"].title(f"{APP_NAME} v{APP_VERSION}")
    except Exception:
        pass

    apply_window_icon(state["root"])

    wire(state)
    if molecule_viewer_host_probe:
        from ui.molecule_viewer_host import open_isolated_molecule_viewer
        state["root"].after_idle(lambda: open_isolated_molecule_viewer(state))
    state["root"].mainloop()


if __name__ == "__main__":
    freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--molecule-viewer-prototype":
        from logic.molecule_viewer_stdio import restore_child_stdio
        restore_child_stdio()
        if len(sys.argv) != 3:
            raise SystemExit("Usage: --molecule-viewer-prototype SCENE_JSON")
        from ui.molecule_viewer_entry import run
        raise SystemExit(run(sys.argv[2]))
    else:
        main(molecule_viewer_host_probe=sys.argv[1:] == ["--molecule-viewer-host-probe"])

'''
When we try to pick out anything by itself, we find it hitched to everything else in the universe.

—John Muir, My First Summer in the Sierra
'''
