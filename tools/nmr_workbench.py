"""Launch the real NMR Workbench without the PCS Analyzer main window.

From the repository root:
    python tools/nmr_workbench.py
    python -m tools.nmr_workbench

The file also runs directly from an IDE with any working directory. Use the
existing Workbench Open/Import/Open session actions to load your own data.
PCS document and structure integrations remain unbound; no host data is mocked.
Application appearance settings are read, but are not saved by this launcher.
"""

from pathlib import Path
import sys


def main() -> None:
    # Direct script execution puts tools/, not the repository, on sys.path.
    repository = str(Path(__file__).resolve().parents[1])
    # Keep it first even if an IDE already added it later: this script's name
    # must not shadow the nmr_workbench package when tools/ is first.
    sys.path.insert(0, repository)

    import tkinter as tk

    from logic.app_settings import load_app_state
    from ui.icons import IconManager
    from ui.style import apply_style
    from nmr_workbench.ui.workbench_window import NMRWorkbenchWindow

    try:
        from tkinterdnd2 import TkinterDnD
    except ImportError:
        root = tk.Tk()
    else:
        try:
            root = TkinterDnD.Tk()
        except (tk.TclError, RuntimeError) as exc:
            print(f"Drag-and-drop unavailable: {exc}. Use Open/Import instead.", file=sys.stderr)
            root = tk.Tk()

    root.withdraw()
    try:
        settings = load_app_state()["app_settings"]
        apply_style(root, variant=settings["theme_variant"],
                    accent=settings["theme_accent"], settings=settings)
        root._icon_manager = IconManager(root)
        window = NMRWorkbenchWindow(root)
        window.protocol("WM_DELETE_WINDOW", window.destroy)

        # Also covers File -> Close or any direct Workbench destruction.
        def on_destroy(event) -> None:
            if event.widget is window:
                root.quit()

        window.bind("<Destroy>", on_destroy, add="+")
        window.lift()
        window.focus_set()
        root.mainloop()
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
