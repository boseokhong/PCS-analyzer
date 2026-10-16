"""Bounded latest-value commands with one scene update awaiting acknowledgement."""

import threading


class ViewerCommandMailbox:
    def __init__(self):
        self._condition = threading.Condition()
        self._ready = False
        self._awaiting_scene = False
        self._scene = None
        self._selection = None
        self._focus = None
        self._shutdown = None
        self._stopped = False

    def ready(self):
        with self._condition:
            self._ready = True
            self._condition.notify_all()

    def scene(self, value):
        with self._condition:
            self._scene = value
            self._selection = None
            self._condition.notify_all()

    def selection(self, value):
        with self._condition:
            self._selection = value
            self._condition.notify_all()

    def focus(self, value):
        with self._condition:
            self._focus = value
            self._condition.notify_all()

    def acknowledge(self):
        with self._condition:
            self._awaiting_scene = False
            self._condition.notify_all()

    def shutdown(self, value):
        with self._condition:
            self._shutdown = value
            self._scene = self._selection = self._focus = None
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def take(self):
        with self._condition:
            while True:
                if self._stopped:
                    return None
                if self._shutdown is not None:
                    result, self._shutdown = self._shutdown, None
                    return ("shutdown", result)
                if self._ready and self._focus is not None:
                    result, self._focus = self._focus, None
                    return ("focus", result)
                if self._ready and not self._awaiting_scene:
                    if self._scene is not None:
                        result, self._scene = self._scene, None
                        self._awaiting_scene = True
                        return ("scene", result)
                    if self._selection is not None:
                        result, self._selection = self._selection, None
                        return ("selection", result)
                self._condition.wait()
