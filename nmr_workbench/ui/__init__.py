"""Tkinter integration for the internal NMR Workbench module."""

from .processing_session import ProcessingSession, ProcessingSessionError
from .view_model import WorkbenchState

__all__ = [
    "NMRWorkbenchWindow",
    "ProcessingSession",
    "ProcessingSessionError",
    "WorkbenchState",
]


def __getattr__(name: str):
    """Load the Tk/Matplotlib window only when the application requests it."""

    if name == "NMRWorkbenchWindow":
        from .workbench_window import NMRWorkbenchWindow

        return NMRWorkbenchWindow
    raise AttributeError(name)

