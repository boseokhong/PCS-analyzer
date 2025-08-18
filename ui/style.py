# ui/style.py

import tkinter as tk
from tkinter import ttk

def apply_style(root, variant="light", accent="blue"):
    used_bootstrap = False
    try:
        import ttkbootstrap as tb
        theme = {
            ("light", "blue"): "flatly",
            ("light", "green"): "minty",
            ("light", "orange"): "journal",
            ("light", "purple"): "yeti",
            ("dark",  "blue"): "cyborg",
            ("dark",  "green"): "darkly",
            ("dark",  "orange"): "superhero",
            ("dark",  "purple"): "solar",
        }.get((variant, accent), "flatly" if variant == "light" else "darkly")
        style = tb.Style(theme)
        used_bootstrap = True
    except Exception:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    # 베이스 색/폰트
    if used_bootstrap:
        root.update_idletasks()
        bg = root.cget("bg") or "#FFFFFF"
        fg = style.lookup("TLabel", "foreground") or "#111111"
    else:
        bg = "#F5F6FA" if variant == "light" else "#1f2125"
        fg = "#111111" if variant == "light" else "#EEEEEE"
        root.configure(bg=bg)

    base_font = ("Segoe UI", 10)
    WHITE = "#FFFFFF"
    GRAY  = "#808080"   # 슬라이더 핸들(활성)용

    style.configure(".", padding=0)
    style.configure("TFrame", padding=6, background=bg)
    style.configure("TLabelframe", background=bg)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)
    style.configure("TLabel", padding=(2, 1), background=bg, foreground=fg)
    style.configure("TNotebook", background=bg, tabposition="r")
    style.configure("TNotebook.Tab", padding=(6, 1), background=bg, foreground=fg)
    style.map("TNotebook.Tab", expand=[("selected", [1,1,1,0])])

    style.configure("TButton", padding=(6, 1))
    # Entry/Combobox: "흰색 배경"
    style.configure("TEntry",
                    padding=(4, 1),
                    fieldbackground=WHITE,
                    background=WHITE)
    style.map("TEntry",
              fieldbackground=[("disabled", "#F0F0F0")])

    style.configure("TCombobox",
                    padding=(4, 1),
                    fieldbackground=WHITE,
                    background=WHITE)

    style.map("TCombobox",
              fieldbackground=[("readonly", WHITE)])


    style.configure("TCheckbutton",
                    background=WHITE,
                    foreground=fg)
    style.configure("TRadiobutton",
                    background=WHITE,
                    foreground=fg)

    # Treeview
    style.configure("Treeview", rowheight=18, font=base_font)
    style.configure("Treeview.Heading", padding=(2,1),
                    font=(base_font[0], base_font[1], "bold"))

    try:
        style.configure("Vertical.TScrollbar", arrowsize=10)
        style.configure("Horizontal.TScrollbar", arrowsize=10)
    except tk.TclError:
        pass

    root.option_add("*Font", base_font)
    root.option_add("*Entry.Font", base_font)
    root.option_add("*TCombobox*Listbox.Font", base_font)

    root.option_add("*Scale.troughColor", WHITE)
    root.option_add("*Scale.activeBackground", GRAY)
    root.option_add("*Scale.highlightThickness", 0)
    root.option_add("*Scale.sliderRelief", "raised")

    # Treeview stripe
    def stripe_treeview(tv: ttk.Treeview):
        tv.tag_configure("oddrow", background="#f6f6f9" if variant=="light" else "#2a2a2e")
        for i, item in enumerate(tv.get_children("")):
            tv.item(item, tags=("oddrow",) if i % 2 else ())

    setattr(root, "_stripe_treeview", stripe_treeview)
    setattr(root, "_app_bg", bg)
    setattr(root, "_app_fg", fg)

    return style
