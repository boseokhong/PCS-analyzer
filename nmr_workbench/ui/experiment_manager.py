"""Dataset-list presentation/selection and ephemeral M13 batch context, no Tk."""
from dataclasses import replace
from ..core import NMRSeries, resolve_temperature
from .view_model import dataset_display_name


class ExperimentManager:
    def __init__(self, state):
        self.state = state
        self.display_ids = []
        self.selected_ids = ()

    def datasets(self):
        by_id = {d.id:d for d in self.state.datasets}
        self.display_ids = [i for i in self.display_ids if i in by_id]
        self.display_ids.extend(d.id for d in self.state.datasets if d.id not in self.display_ids)
        self.selected_ids = tuple(i for i in self.selected_ids if i in by_id)
        return tuple(by_id[i] for i in self.display_ids)

    def select(self, ids):
        self.datasets()
        requested = set(ids)
        if requested - set(self.display_ids):
            raise ValueError("Unknown experiment selection")
        self.selected_ids = tuple(i for i in self.display_ids if i in requested)

    def reorder(self, moved_ids, before_id=None):
        self.datasets()
        moved = [i for i in self.display_ids if i in set(moved_ids)]
        if before_id in moved or not moved:
            return
        remaining = [i for i in self.display_ids if i not in moved]
        index = remaining.index(before_id) if before_id is not None else len(remaining)
        self.display_ids = remaining[:index] + moved + remaining[index:]

    def sort(self, order):
        datasets = self.datasets()
        if order == "Import order":
            self.display_ids = [d.id for d in self.state.datasets]
            return
        def key(dataset):
            if order == "Name":
                return (False, dataset_display_name(dataset).casefold())
            if order == "Temperature":
                value = resolve_temperature(dataset.conditions)
                return (value.status != "valid", value.temperature_k or 0.)
            if order == "Acquisition time":
                value = dataset.conditions.acquisition_date
                return (value is None, str(value or ""))
            raise ValueError("Unknown experiment ordering")
        self.display_ids = [d.id for d in sorted(datasets, key=key)]

    def rename(self, dataset_id, label):
        if not label or not label.strip():
            raise ValueError("Experiment label must not be blank")
        dataset = next(d for d in self.datasets() if d.id == dataset_id)
        dataset.vendor_metadata["display_name"] = label.strip()

    def remove(self, ids):
        ids = set(ids)
        removed = tuple(d for d in self.state.datasets if d.id in ids)
        for series in self.state.series:
            for member in tuple(series.members):
                if member.dataset_id in ids:
                    series.remove_member(member.id)  # Explicit stale links retained.
        self.state.datasets[:] = [d for d in self.state.datasets if d.id not in ids]
        if self.state.selected_dataset_id in ids:
            self.state.selected_dataset_id = self.state.datasets[0].id if self.state.datasets else None
        self.state._sync_series_member()
        self.datasets()
        return removed

    def batch_context(self):
        selected = set(self.selected_ids)
        context = NMRSeries("Selected experiments (temporary batch)")
        active = next((s for s in self.state.series if s.id == self.state.selected_series_id), None)
        for dataset in self.datasets():
            if dataset.id not in selected:
                continue
            member = next((m for m in active.members if m.dataset is dataset), None) if active else None
            added = context.add_member(dataset)
            if member is not None:
                context.replace_member(replace(added, included=member.included, label=member.label))
        return context
