"""GUI-independent child scene/selection state, not an authoritative document."""

from pathlib import Path

from logic.molecule_viewer_protocol import event_for, matches_scene, read_scene


class ViewerSession:
    def __init__(self, scene, runtime_path):
        self.scene = scene
        self.runtime_path = Path(runtime_path).resolve()
        self.selected_refs = ()
        self.host_selection_revision = 0
        self.pick_sequence = 0

    def replacement(self, command):
        if command.get("session_id") != self.scene["session_id"]:
            raise ValueError("Invalid viewer session")
        path = Path(command["path"]).resolve()
        if path.parent != self.runtime_path or not path.name.startswith("update-"):
            raise ValueError("Scene update must be an owned snapshot file")
        scene = read_scene(path)
        if not matches_scene(command, scene) or scene["session_id"] != self.scene["session_id"]:
            raise ValueError("Scene update identity mismatch")
        if scene.get("scene_generation", 0) <= self.scene.get("scene_generation", 0):
            return None
        return scene

    def commit(self, scene):
        self.scene = scene
        self.selected_refs = ()
        self.host_selection_revision = 0
        return event_for(scene, "scene_applied")

    def select(self, command):
        if not matches_scene(command, self.scene):
            return None
        revision = command.get("selection_revision")
        if type(revision) is not int or revision <= self.host_selection_revision:
            return None
        refs = command.get("ref_ids")
        if (not isinstance(refs, list) or len(refs) > 1
                or any(type(r) is not int or r not in self.scene["ref_ids"] for r in refs)):
            raise ValueError("Invalid display selection")
        self.selected_refs = tuple(refs)
        self.host_selection_revision = revision
        # This acknowledgement is not a viewer-origin selection event.
        return event_for(self.scene, "selection_applied", selection_revision=revision,
                         ref_ids=list(self.selected_refs))

    def pick(self, ref):
        if type(ref) is not int or ref not in self.scene["ref_ids"]:
            return None
        self.selected_refs = (ref,)
        self.pick_sequence += 1
        return event_for(self.scene, "selection", ref_ids=[ref], origin="viewer",
                         pick_sequence=self.pick_sequence,
                         host_selection_revision=self.host_selection_revision)
