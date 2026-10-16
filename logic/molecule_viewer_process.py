"""Asynchronous ownership of one isolated molecular-viewer process.

No GUI imports or DPI calls belong in the parent launch path. The host must call
poll() regularly from its event loop and request_close() on confirmed shutdown.
The Tk host adapter consumes this private API for the default molecular viewer.
Scientific state and project persistence remain owned by the host.
"""

import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from logic.molecule_viewer_protocol import (
    EVENT_PREFIX, MAX_EVENT_BYTES, MAX_SCENE_BYTES, event_for, matches_scene, validate_scene,
)
from logic.molecule_viewer_mailbox import ViewerCommandMailbox

VIEWER_ROLE = "--molecule-viewer-prototype"


def viewer_command(scene_path, *, executable=None, frozen=None):
    executable = str(executable or sys.executable)
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    args = [executable]
    if not frozen:
        args += [str(Path(__file__).resolve().parents[1] / "main.py")]
    return args + [VIEWER_ROLE, str(Path(scene_path).resolve())]


def child_environment():
    env = os.environ.copy()
    env["QT_API"] = "pyside6"
    env["PYTHONUNBUFFERED"] = "1"
    # Do not modify host environment, DPI awareness or existing scale variables.
    return env


class MoleculeViewerProcess:
    """Nonblocking lifecycle API; no callbacks execute on reader/writer threads."""

    def __init__(self, scene, *, startup_timeout=30.0, close_timeout=2.0,
                 command_factory=viewer_command):
        if (not math.isfinite(startup_timeout) or not math.isfinite(close_timeout)
                or startup_timeout <= 0 or close_timeout <= 0):
            raise ValueError("Lifecycle timeouts must be positive")
        # Copy and JSON-normalize without retaining host arrays/widgets/state.
        payload = json.dumps(dict(scene, session_id=uuid.uuid4().hex, scene_generation=0), allow_nan=False)
        if len(payload.encode("utf-8")) > MAX_SCENE_BYTES:
            raise ValueError("Scene exceeds prototype message budget")
        self.scene = validate_scene(json.loads(payload))
        self._initial_scene = self._rendered_scene = self.scene
        self._inflight_scene = None
        self._update_timeout = startup_timeout
        self._selection_revision = 0
        self._pick_sequence = 0
        self._legacy_sequence = 0
        self._display_sequence = 0
        self.state = "starting"
        self.error = None
        self.returncode = None
        self._startup_deadline = time.monotonic() + startup_timeout
        self._close_timeout = close_timeout
        self._close_deadline = None
        self._events = queue.Queue(maxsize=64)
        self._commands = ViewerCommandMailbox()
        self._reader_done = threading.Event()
        self.cleanup_complete = threading.Event()
        self.cleanup_errors = []
        self.shutdown_write_error = None
        self._ready_seen = False
        self._expected_close = False
        self._diagnostic_tail = b""
        self._temporary = tempfile.TemporaryDirectory(prefix="pcs-molecule-viewer-")
        self.runtime_path = Path(self._temporary.name)
        self.process = None
        self._disposed = False
        try:
            scene_path = self.runtime_path / "scene.json"
            scene_path.write_text(payload, encoding="utf-8")
            self.process = subprocess.Popen(
                command_factory(scene_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=child_environment(), close_fds=True,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception:
            self._temporary.cleanup()
            raise
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._writer = threading.Thread(target=self._write_commands, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_diagnostics, daemon=True)
        self._reader.start()
        self._writer.start()
        self._stderr_reader.start()

    @property
    def diagnostics(self):
        """Bounded stderr tail; available even after owned files are removed."""
        return self._diagnostic_tail.decode("utf-8", errors="replace")

    def _read_diagnostics(self):
        try:
            while True:
                chunk = self.process.stderr.read(1024)
                if not chunk:
                    return
                self._diagnostic_tail = (self._diagnostic_tail + chunk)[-MAX_EVENT_BYTES:]
        except OSError as exc:
            self._diagnostic_tail = str(exc).encode("utf-8")[-MAX_EVENT_BYTES:]

    def _enqueue(self, event):
        try:
            self._events.put_nowait(event)
        except queue.Full:
            # A peer flooding the bounded event queue is a transport failure.
            try:
                self.process.terminate()
            except OSError:
                # The owned process may have exited between queue overflow and termination.
                self._reader_done.set()

    def _read_events(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_EVENT_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_EVENT_BYTES:
                    self._enqueue({"type": "transport_error", "message": "Oversized viewer output"})
                    break
                if line.startswith(EVENT_PREFIX.encode("ascii")):
                    event = json.loads(line[len(EVENT_PREFIX):])
                    if not isinstance(event, dict):
                        raise ValueError("Viewer event must be an object")
                    self._enqueue(event)
        except (OSError, ValueError) as exc:
            self._enqueue({"type": "transport_error", "message": str(exc)[:1024]})
        finally:
            self._enqueue({"type": "channel_closed"})
            self._reader_done.set()

    def _write_commands(self):
        while True:
            item = self._commands.take()
            if item is None:
                return
            kind, command = item
            try:
                if kind == "scene":
                    path = self.runtime_path / f"update-{command['scene_generation']}.json"
                    path.write_text(json.dumps(command, allow_nan=False), encoding="utf-8")
                    self._inflight_scene = (command, path, time.monotonic())
                    command = event_for(command, "scene", path=str(path))
                data = json.dumps(command, allow_nan=False).encode("utf-8") + b"\n"
                while data:
                    count = self.process.stdin.write(data)
                    if not count:
                        raise OSError("Viewer command pipe closed")
                    data = data[count:]
            except (OSError, ValueError) as exc:
                if kind == "shutdown":
                    self.shutdown_write_error = str(exc)[:1024]
                else:
                    self._enqueue({"type": "transport_error", "message": str(exc)[:1024]})
                return
            if kind == "shutdown":
                # The command reader can already be waiting for its next line.
                # Deliver EOF from the writer thread, never wait for child exit
                # before closing this pipe (that creates a shutdown cycle).
                try:
                    self.process.stdin.close()
                except OSError as exc:
                    self.shutdown_write_error = str(exc)[:1024]
                return

    def update_scene(self, scene):
        """Queue a copied snapshot; superseded pending scenes never enter the pipe."""
        if self._close_deadline is not None or self._disposed:
            raise RuntimeError("Viewer is closing")
        payload = json.dumps(dict(scene, session_id=self.scene["session_id"],
                                  scene_generation=self.scene["scene_generation"] + 1), allow_nan=False)
        if len(payload.encode("utf-8")) > MAX_SCENE_BYTES:
            raise ValueError("Scene exceeds prototype message budget")
        self.scene = validate_scene(json.loads(payload))
        self._commands.scene(self.scene)
        self.set_selection([])

    def request_focus(self):
        """Coalesce a child-only foreground request; no native APIs in the host."""
        if self._close_deadline is None and not self._disposed:
            self._commands.focus(event_for(self.scene, "focus"))

    def set_selection(self, refs):
        """Display the host's primary Ref only; a hidden primary stays hidden."""
        if self._close_deadline is not None or self._disposed:
            raise RuntimeError("Viewer is closing")
        refs = list(refs)
        if any(type(ref) is not int for ref in refs):
            raise ValueError("Selection must contain integer Ref IDs")
        primary = refs[:1] if refs and refs[0] in self.scene["ref_ids"] else []
        self._selection_revision += 1
        self._commands.selection(event_for(self.scene, "selection", ref_ids=primary,
                                          selection_revision=self._selection_revision))

    def request_close(self):
        """Request shutdown after the caller has confirmed host/viewer closure."""
        self._expected_close = True
        self._begin_close()

    def _begin_close(self):
        if self._disposed or self._close_deadline is not None:
            return
        self._close_deadline = time.monotonic() + self._close_timeout
        if self.error is None:
            self.state = "closing"
        self._commands.shutdown({"type": "shutdown", "session_id": self.scene["session_id"]})

    def poll(self):
        """Drain lifecycle, scene, selection and export requests without waiting.

        Call this and mutation methods from the host event-loop thread. Returned
        selections are requests for the host adapter, never direct state edits.
        """
        if self._disposed:
            return []
        events = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("type")
            if kind == "display_change":
                from logic.molecule_viewer_display import validate_display_change
                sequence = event.get("display_sequence")
                if (self.state == "ready" and "display_controls" in self.scene
                        and matches_scene(event, self.scene) and type(sequence) is int
                        and sequence > self._display_sequence
                        and validate_display_change(event.get("key"), event.get("value"))):
                    self._display_sequence = sequence
                    events.append(event)
                continue
            if kind == "open_matplotlib":
                sequence = event.get("legacy_sequence")
                if (self.state == "ready" and matches_scene(event, self.scene)
                        and type(sequence) is int
                        and sequence > self._legacy_sequence):
                    self._legacy_sequence = sequence
                    events.append(event)
                continue
            if kind in ("channel_closed", "transport_error"):
                if kind == "transport_error":
                    if self.error is None:
                        self.error = event["message"]
                elif not self._ready_seen and not self._expected_close and self.error is None:
                    self.error = "Viewer disconnected before readiness"
                self._begin_close()
                continue
            if kind == "scene_applied":
                inflight = self._inflight_scene
                if inflight is None or not matches_scene(event, inflight[0]):
                    self.error = "Invalid scene acknowledgement"
                    self._begin_close()
                    continue
                self._rendered_scene = inflight[0]
                try:
                    inflight[1].unlink()
                except OSError as exc:
                    self.error = f"Could not retire viewer snapshot: {exc}"
                    self._begin_close()
                self._inflight_scene = None
                self._commands.acknowledge()
                if matches_scene(event, self.scene):
                    events.append(event)
                continue
            if kind in ("selection", "selection_applied"):
                # Expected stale replies from old documents/scenes are discarded.
                if not matches_scene(event, self.scene):
                    continue
                if kind == "selection_applied":
                    if (type(event.get("selection_revision")) is int
                            and event["selection_revision"] == self._selection_revision):
                        events.append(event)
                    continue
                refs = event.get("ref_ids")
                sequence = event.get("pick_sequence")
                if (event.get("origin") != "viewer" or not isinstance(refs, list)
                        or len(refs) != 1 or type(refs[0]) is not int
                        or refs[0] not in self.scene["ref_ids"]
                        or type(sequence) is not int or sequence <= self._pick_sequence
                        or type(event.get("host_selection_revision")) is not int
                        or event["host_selection_revision"] != self._selection_revision):
                    continue
                self._pick_sequence = sequence
                events.append(event)
                continue
            identities = [self._initial_scene] if kind == "ready" else [self._rendered_scene]
            if self._inflight_scene is not None and kind != "ready":
                identities.append(self._inflight_scene[0])
            if not any(matches_scene(event, value) for value in identities):
                self.error = "Viewer replied with an obsolete or invalid identity"
                self._begin_close()
                continue
            if kind == "ready" and self._close_deadline is None:
                self._ready_seen = True
                self.state = "ready"
                self._commands.ready()
            elif kind == "error":
                self.error = str(event.get("message", "Viewer startup failed"))[:1024]
                self._begin_close()
            elif kind == "closed":
                self._begin_close()
            elif kind != "ready":
                self.error = "Unexpected viewer lifecycle event"
                self._begin_close()
            events.append(event)
        now = time.monotonic()
        if self.state == "starting" and now >= self._startup_deadline:
            self.error = "Viewer startup timed out"
            self._begin_close()
        if (self._inflight_scene is not None and self._close_deadline is None
                and now - self._inflight_scene[2] >= self._update_timeout):
            self.error = "Viewer scene update timed out"
            self._begin_close()
        code = self.process.poll()
        if code is None and self._close_deadline is not None and now >= self._close_deadline:
            # Only our owned child is killed; no process-name/global termination.
            if self.error is None:
                self.error = "Viewer shutdown timed out"
            try:
                self.process.kill()
            except OSError as exc:
                if self.process.poll() is None:
                    self.error = f"Could not stop owned viewer: {str(exc)[:768]}"
            code = self.process.poll()
        if code is not None:
            # A process can exit before its final buffered event reaches the reader.
            # Drain that event on the next poll rather than losing the diagnostic.
            if not self._reader_done.is_set() or not self._events.empty():
                return events
            self.returncode = code
            if code != 0 and self.error is None:
                self.error = f"Viewer exited with status {code}"
            elif code == 0 and not self._ready_seen and not self._expected_close and self.error is None:
                self.error = "Viewer exited before readiness"
            self.state = "failed" if self.error is not None else "closed"
            self._dispose()
        return events

    def _dispose(self):
        # The process is already reaped by poll(). Joining background pipe readers
        # is a cleanup step in a separate thread, never a Tk/Qt event-loop wait.
        self._disposed = True
        self._commands.stop()
        threading.Thread(target=self._cleanup, daemon=True).start()

    def _cleanup(self):
        self._reader.join()
        self._writer.join()
        self._stderr_reader.join()
        # A broken Windows pipe must not prevent closing the other handles or
        # leave stderr locked. Unbuffered pipes also avoid flushing failed writes.
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except OSError as exc:
                self.cleanup_errors.append(str(exc))
        try:
            self._temporary.cleanup()
        except OSError as exc:
            self.cleanup_errors.append(f"Temporary viewer files not removed: {exc}")
        finally:
            self.cleanup_complete.set()
