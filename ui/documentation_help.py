"""Open the repository's public manual with the user's default browser."""
import webbrowser

DOCUMENTATION_URL = "https://boseokhong.github.io/PCS-analyzer/"


def open_documentation(state):
    try:
        if not webbrowser.open_new_tab(DOCUMENTATION_URL):
            raise RuntimeError("No browser accepted the request.")
    except Exception as exc:
        state["messagebox"].showerror("Documentation", f"Could not open documentation.\n{DOCUMENTATION_URL}\n\n{exc}",
                                       parent=state["root"])
        return False
    return True
