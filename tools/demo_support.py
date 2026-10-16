"""Small CLI/file helpers for reproducible source examples; no UI or numerical code."""
from pathlib import Path
import argparse
from dataclasses import dataclass


def generation_args(default, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, default=Path(default), help="Generated data directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace named generated files")
    return parser.parse_args()


def prepare_output(destination, filenames, *, overwrite=False):
    """Preflight all generated files before writing; never remove directories."""
    destination = Path(destination).resolve()
    paths = [destination / name for name in filenames]
    if any(not p.resolve().is_relative_to(destination) for p in paths):
        raise ValueError("Generated file paths must stay within the output directory")
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Generated files already exist; use --overwrite: {existing[0]}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def detached_standard_state(state):
    """Bridge the loader's plain entry value to the legacy fitter's numeric fallback."""
    result = dict(state)
    entry = result.pop("tensor_entry", None)
    if entry is not None:
        value = entry.get() if hasattr(entry, "get") else entry
        if str(value).strip():
            result["tensor"] = float(value)
    return result


@dataclass(frozen=True)
class _ReadValue:
    """Only the legacy serializer's read protocol; never creates a Tk variable."""
    value: object

    def get(self):
        return self.value


def demo_project_text(state):
    """Serialize explicit teaching controls via the existing project writer."""
    from logic.project_io import build_pcsp_text
    values = dict(state)
    for key, value in state.items():
        if key.endswith('_var') and not hasattr(value, 'get'):
            values[key] = _ReadValue(value)
    if 'tensor' in state and 'tensor_entry' not in state:
        values['tensor_entry'] = _ReadValue(state['tensor'])
    return '\n'.join(line.rstrip() for line in build_pcsp_text(values).splitlines())+'\n'
