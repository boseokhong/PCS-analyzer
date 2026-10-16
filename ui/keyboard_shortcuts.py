"""One catalog for menu accelerators, main-window bindings and Help reference."""
SHORTCUTS = (
    ("Documents", "New PCS document", "app.file.new_document", "Ctrl+N", ("<Control-n>", "<Control-N>")),
    ("Documents", "Open PCS project", "app.file.open_project", "Ctrl+O", ("<Control-o>", "<Control-O>")),
    ("Documents", "Save PCS project", "app.file.save_project", "Ctrl+S", ("<Control-s>", "<Control-S>")),
    ("Documents", "Save PCS project as", "app.file.save_project_as", "Ctrl+Shift+S", ("<Control-Shift-s>", "<Control-Shift-S>")),
    ("Documents", "Close current document", "app.file.close_document", "Ctrl+W", ("<Control-w>", "<Control-W>")),
    ("Document navigation", "Next document", "app.file.next_document", "Ctrl+Tab", ("<Control-Tab>",)),
    ("Document navigation", "Previous document", "app.file.previous_document", "Ctrl+Shift+Tab", ("<Control-Shift-Tab>", "<Control-ISO_Left_Tab>", "<Control-Shift-ISO_Left_Tab>")),
    ("Edit history", "Undo document edit", "history.undo", "Ctrl+Z", ("<Control-z>",)),
    ("Edit history", "Redo document edit", "history.redo", "Ctrl+Y", ("<Control-y>",)),
    ("Edit history", "Redo document edit (alternative)", "history.redo", "Ctrl+Shift+Z", ("<Control-Shift-Z>",)),
    ("Help", "Documentation", "app.help.documentation", "F1", ("<F1>",)),
)
MENU_SHORTCUTS = {command: keys for _, _, command, keys, _ in SHORTCUTS if not command.startswith("history.")}


def navigate_document(state, direction):
    manager = state["document_manager"]
    ids = manager.document_ids
    if len(ids) < 2:
        return False
    index = ids.index(manager.active_document_id)
    return manager.set_active_document(ids[(index + direction) % len(ids)])


def install_main_shortcuts(state):
    """Handle document commands before Notebook/Text class bindings, within main only."""
    root = state["root"]
    registry = state["command_registry"]
    tag = f"PCSMainShortcuts:{root}"
    def invoke(event, command):
        if event.widget.winfo_toplevel() is not root or root.grab_current() is not None:
            return None
        registry.invoke(command)
        return "break"
    for _, _, command, _, sequences in SHORTCUTS:
        if command.startswith("history."):
            continue  # Existing widget-local Undo/Redo priority remains authoritative.
        for sequence in sequences:
            root.bind_class(tag, sequence, lambda event, c=command: invoke(event, c))
    def attach(widget):
        if widget.winfo_toplevel() is not root:
            return
        tags = widget.bindtags()
        if tag not in tags:
            widget.bindtags((tag,) + tags)
        for child in widget.winfo_children():
            attach(child)
    attach(root)
    root.bind("<Map>", lambda event: attach(event.widget), add="+")


def open_shortcuts_window(state):
    import tkinter as tk
    from tkinter import ttk
    from tkinter import font as tkfont
    from ui.style import get_app_fonts
    root = state["root"]
    existing = state.get("keyboard_shortcuts_window")
    if existing is not None and existing.winfo_exists():
        existing.lift()
        existing.focus_force()
        return existing
    win = tk.Toplevel(root)
    win.withdraw()
    win.title("Keyboard Shortcuts")
    win.transient(root)
    state["keyboard_shortcuts_window"] = win
    body = ttk.Frame(win, padding=14)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Keyboard Shortcuts", font=get_app_fonts(state)["title"]).pack(anchor="w", pady=(0,8))
    table_frame = ttk.Frame(body)
    table_frame.pack(fill="both", expand=True)
    table = ttk.Treeview(table_frame, columns=("keys",), show="tree headings", height=16)
    table.heading("#0", text="Action")
    table.heading("keys", text="Shortcut")
    font = tkfont.Font(root=root, font=get_app_fonts(state)["ui"])
    action_width = max(300, max(font.measure(action) for _, action, _, _, _ in SHORTCUTS) + 44)
    keys_width = max(180, max(font.measure(keys) for _, _, _, keys, _ in SHORTCUTS) + 24)
    table.column("#0", width=action_width, minwidth=200)
    table.column("keys", width=keys_width, minwidth=145, stretch=False)
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    table.pack(side="left", fill="both", expand=True)
    groups = {}
    for group, action, _, keys, _ in SHORTCUTS:
        if group not in groups:
            groups[group] = table.insert("", "end", text=group, open=True)
        table.insert(groups[group], "end", text=action, values=(keys,))
    ttk.Label(body, text="Document shortcuts apply in the main window. Entry/text controls keep local Undo/Redo.",
              wraplength=480, justify="left").pack(fill="x", pady=(10,8))
    ttk.Button(body, text="Close", command=win.destroy).pack(anchor="e")
    win.bind("<Escape>", lambda event: win.destroy())
    win.update_idletasks()
    width, height = win.winfo_reqwidth(), win.winfo_reqheight()
    x = root.winfo_rootx() + (root.winfo_width() - width) // 2
    y = root.winfo_rooty() + (root.winfo_height() - height) // 2
    win.geometry(f"{width}x{height}{x:+d}{y:+d}")
    win.minsize(width, min(height, 320))
    win.deiconify()
    return win
