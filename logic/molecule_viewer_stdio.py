"""Recover inherited pipes in a Windows windowed child role only.

PyInstaller console=False may leave sys.std* as None even when Popen supplied
OS pipe handles. Duplicate those handles rather than opening a console or
changing the host. No GUI or DPI initialization belongs in this module.
"""

import io
import os
import sys


def iter_command_lines(fd, max_bytes):
    """Read bounded commands without holding Python's buffered stdin lock.

    This reader runs only in the child pipe thread. The descriptor is borrowed;
    EOF is delivered by the host closing its command pipe after shutdown.
    """
    pending = b""
    while True:
        chunk = os.read(fd, max_bytes + 1)
        if not chunk:
            if pending:
                raise ValueError("Incomplete viewer command")
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            line += b"\n"
            if len(line) > max_bytes:
                raise ValueError("Oversized viewer command")
            yield line
        if len(pending) >= max_bytes:
            raise ValueError("Oversized viewer command")


def _windows_stream(handle_id, *, reading):
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel.GetStdHandle.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.DuplicateHandle.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
    ]
    kernel.DuplicateHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    inherited = kernel.GetStdHandle(handle_id)
    if inherited in (None, ctypes.c_void_p(-1).value):
        raise OSError("Molecular viewer child requires inherited standard pipes")
    duplicate = wintypes.HANDLE()
    current = kernel.GetCurrentProcess()
    if not kernel.DuplicateHandle(current, inherited, current, ctypes.byref(duplicate), 0, False, 2):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        flags = os.O_BINARY | (os.O_RDONLY if reading else os.O_WRONLY)
        fd = msvcrt.open_osfhandle(duplicate.value, flags)
    except OSError:
        kernel.CloseHandle(duplicate)
        raise
    try:
        binary = io.FileIO(fd, mode="rb" if reading else "wb", closefd=True)
    except Exception:
        os.close(fd)
        raise
    return io.TextIOWrapper(binary, encoding="utf-8", errors="replace", write_through=True)


def restore_child_stdio():
    """Called exclusively after dispatch to the private viewer child role."""
    for name, handle_id, reading in (
        ("stderr", -12, False), ("stdout", -11, False), ("stdin", -10, True)
    ):
        if getattr(sys, name) is None:
            if os.name != "nt":
                raise OSError("Viewer standard pipes are unavailable")
            setattr(sys, name, _windows_stream(handle_id, reading=reading))
