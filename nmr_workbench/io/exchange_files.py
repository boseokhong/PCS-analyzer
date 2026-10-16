"""Atomic text exports: failed writes leave an existing user file intact."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_text_export(path, *, encoding="utf-8", newline=None):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline=newline,
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
