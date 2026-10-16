"""Reusable compact Ribbon presentation widgets for PCS Analyzer."""

from .ribbon_bar import RibbonBar, RibbonTabStrip
from .ribbon_button import RibbonButton
from .ribbon_checkbutton import RibbonCheckbutton
from .context_controller import ContextRibbonController
from .ribbon_group import RibbonGroup
from .ribbon_page import RibbonPage
from .ribbon_toggle_button import RibbonToggleButton
from .icon_service import (
    RibbonIconService,
    nearest_packaged_icon_size,
    tk_scale_factor,
)

__all__ = [
    "RibbonBar",
    "RibbonTabStrip",
    "RibbonPage",
    "RibbonGroup",
    "RibbonButton",
    "RibbonCheckbutton",
    "RibbonToggleButton",
    "ContextRibbonController",
    "RibbonIconService",
    "nearest_packaged_icon_size",
    "tk_scale_factor",
]
