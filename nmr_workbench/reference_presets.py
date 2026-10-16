from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


_RESOURCE_PATH = Path(__file__).resolve().parent / "resources" / "reference_presets.json"


class ReferencePresetError(ValueError):
    """Raised when the reference-preset resource is malformed or queried incorrectly."""


@dataclass(frozen=True)
class ReferenceSource:
    id: str
    citation: str
    url: str | None = None
    doi: str | None = None
    year: int | None = None
    publisher: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ReferenceResonance:
    id: str
    label: str
    ppm: float
    source_id: str
    temperature_K: float | None = None
    multiplicity: str | None = None
    exchangeable: bool = False
    recommended_for_direct_referencing: bool = True
    verification: tuple[dict[str, Any], ...] = ()
    notes: str | None = None


@dataclass(frozen=True)
class ReferencePreset:
    id: str
    display_name: str
    category: str
    aliases: tuple[str, ...]
    resonances: dict[str, tuple[ReferenceResonance, ...]]


def _normalize_alias(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("−", "-")
        .replace("–", "-")
        .replace("_", "")
        .replace(" ", "")
    )


def _parse_source(source_id: str, payload: dict[str, Any]) -> ReferenceSource:
    return ReferenceSource(
        id=source_id,
        citation=str(payload["citation"]),
        url=payload.get("url"),
        doi=payload.get("doi"),
        year=payload.get("year"),
        publisher=payload.get("publisher"),
        notes=payload.get("notes"),
    )


def _parse_resonance(payload: dict[str, Any]) -> ReferenceResonance:
    conditions = payload.get("conditions") or {}
    return ReferenceResonance(
        id=str(payload["id"]),
        label=str(payload["label"]),
        ppm=float(payload["ppm"]),
        source_id=str(payload["source_id"]),
        temperature_K=(
            None if conditions.get("temperature_K") is None
            else float(conditions["temperature_K"])
        ),
        multiplicity=payload.get("multiplicity"),
        exchangeable=bool(payload.get("exchangeable", False)),
        recommended_for_direct_referencing=bool(
            payload.get("recommended_for_direct_referencing", True)
        ),
        verification=tuple(payload.get("verification") or ()),
        notes=payload.get("notes"),
    )


def _parse_preset(
    preset_id: str,
    payload: dict[str, Any],
    *,
    expected_category: str,
    known_source_ids: set[str],
) -> ReferencePreset:
    category = str(payload.get("category", expected_category))
    if category != expected_category:
        raise ReferencePresetError(
            f"Preset {preset_id!r} has category {category!r}, expected {expected_category!r}."
        )

    raw_resonances = payload.get("resonances")
    if not isinstance(raw_resonances, dict):
        raise ReferencePresetError(f"Preset {preset_id!r} is missing a resonances mapping.")

    parsed: dict[str, tuple[ReferenceResonance, ...]] = {}
    seen_ids: set[tuple[str, str]] = set()

    for nucleus, rows in raw_resonances.items():
        if nucleus not in {"1H", "13C"}:
            raise ReferencePresetError(
                f"Preset {preset_id!r} contains unsupported nucleus {nucleus!r}."
            )
        if not isinstance(rows, list):
            raise ReferencePresetError(
                f"Preset {preset_id!r}/{nucleus} resonances must be a list."
            )
        out = []
        for row in rows:
            resonance = _parse_resonance(row)
            key = (nucleus, resonance.id)
            if key in seen_ids:
                raise ReferencePresetError(
                    f"Duplicate resonance id {resonance.id!r} for {preset_id!r}/{nucleus}."
                )
            seen_ids.add(key)
            if resonance.source_id not in known_source_ids:
                raise ReferencePresetError(
                    f"Unknown source_id {resonance.source_id!r} for "
                    f"{preset_id!r}/{nucleus}/{resonance.id!r}."
                )
            for verification in resonance.verification:
                verify_source = verification.get("source_id")
                if verify_source and verify_source not in known_source_ids:
                    raise ReferencePresetError(
                        f"Unknown verification source_id {verify_source!r} for "
                        f"{preset_id!r}/{nucleus}/{resonance.id!r}."
                    )
            out.append(resonance)
        parsed[nucleus] = tuple(out)

    return ReferencePreset(
        id=preset_id,
        display_name=str(payload["display_name"]),
        category=category,
        aliases=tuple(str(x) for x in payload.get("aliases", ())),
        resonances=parsed,
    )


class ReferencePresetLibrary:
    def __init__(
        self,
        *,
        presets: Iterable[ReferencePreset],
        sources: dict[str, ReferenceSource],
    ) -> None:
        self._presets = tuple(presets)
        self._sources = dict(sources)
        self._by_id = {preset.id: preset for preset in self._presets}
        if len(self._by_id) != len(self._presets):
            raise ReferencePresetError("Duplicate preset IDs.")

        self._alias_index: dict[str, ReferencePreset] = {}
        for preset in self._presets:
            for alias in (preset.id, preset.display_name, *preset.aliases):
                normalized = _normalize_alias(alias)
                existing = self._alias_index.get(normalized)
                if existing is not None and existing.id != preset.id:
                    raise ReferencePresetError(
                        f"Ambiguous reference preset alias {alias!r}: "
                        f"{existing.id!r} vs {preset.id!r}."
                    )
                self._alias_index[normalized] = preset

    @property
    def presets(self) -> tuple[ReferencePreset, ...]:
        return self._presets

    @property
    def sources(self) -> dict[str, ReferenceSource]:
        return dict(self._sources)

    def get(self, preset_id: str) -> ReferencePreset:
        try:
            return self._by_id[preset_id]
        except KeyError as exc:
            raise ReferencePresetError(f"Unknown reference preset: {preset_id}") from exc

    def resolve_alias(self, value: str) -> ReferencePreset | None:
        if not value:
            return None
        return self._alias_index.get(_normalize_alias(value))

    def presets_for_category(self, category: str) -> tuple[ReferencePreset, ...]:
        return tuple(p for p in self._presets if p.category == category)

    def presets_for_nucleus(
        self,
        nucleus: str,
        *,
        category: str | None = None,
    ) -> tuple[ReferencePreset, ...]:
        result = []
        for preset in self._presets:
            if category is not None and preset.category != category:
                continue
            if preset.resonances.get(nucleus):
                result.append(preset)
        return tuple(result)

    def resonances_for(
        self,
        preset_id: str,
        nucleus: str,
        *,
        include_nonpreferred: bool = True,
    ) -> tuple[ReferenceResonance, ...]:
        rows = self.get(preset_id).resonances.get(nucleus, ())
        if include_nonpreferred:
            return rows
        return tuple(r for r in rows if r.recommended_for_direct_referencing)

    def sorted_resonances_for(
        self,
        preset_id: str,
        nucleus: str,
        *,
        observed_ppm: float | None = None,
        include_nonpreferred: bool = True,
    ) -> tuple[ReferenceResonance, ...]:
        rows = list(
            self.resonances_for(
                preset_id,
                nucleus,
                include_nonpreferred=include_nonpreferred,
            )
        )
        if observed_ppm is not None:
            rows.sort(key=lambda r: abs(r.ppm - float(observed_ppm)))
        return tuple(rows)

    def source_for(self, resonance: ReferenceResonance) -> ReferenceSource:
        return self._sources[resonance.source_id]


def load_reference_preset_library(
    path: str | Path | None = None,
) -> ReferencePresetLibrary:
    resource_path = Path(path) if path is not None else _RESOURCE_PATH
    try:
        payload = json.loads(resource_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferencePresetError(
            f"Could not load reference preset resource: {resource_path}"
        ) from exc

    if payload.get("schema_version") != 1:
        raise ReferencePresetError(
            f"Unsupported reference-preset schema version: {payload.get('schema_version')!r}"
        )

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ReferencePresetError("Reference preset resource has no sources.")

    sources = {
        source_id: _parse_source(source_id, source_payload)
        for source_id, source_payload in raw_sources.items()
    }
    known_source_ids = set(sources)

    presets: list[ReferencePreset] = []
    for preset_id, preset_payload in (payload.get("residual_solvents") or {}).items():
        presets.append(
            _parse_preset(
                preset_id,
                preset_payload,
                expected_category="residual_solvent",
                known_source_ids=known_source_ids,
            )
        )
    for preset_id, preset_payload in (payload.get("reference_standards") or {}).items():
        presets.append(
            _parse_preset(
                preset_id,
                preset_payload,
                expected_category="reference_standard",
                known_source_ids=known_source_ids,
            )
        )

    return ReferencePresetLibrary(presets=presets, sources=sources)
