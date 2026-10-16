from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from plugin_system.plugin_version import (
    CompatibilityResult,
    PluginVersionError,
    check_app_compatibility,
    parse_version,
)


PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
REQUIRED_MANIFEST_FIELDS = ("id", "name", "version", "entry")


class ManifestValidationError(ValueError):
    """Raised when plugin manifest metadata is malformed or unsafe."""


def _entry_parts(entry: object) -> tuple[str, ...]:
    if not isinstance(entry, str) or not entry.strip():
        raise ManifestValidationError("entry must be a non-empty string.")
    if "\x00" in entry:
        raise ManifestValidationError("entry must not contain NUL characters.")

    portable_entry = entry.replace("\\", "/")
    if portable_entry.startswith("/") or re.match(r"^[A-Za-z]:", portable_entry):
        raise ManifestValidationError("entry must be a relative path.")

    parts = tuple(part for part in portable_entry.split("/") if part not in ("", "."))
    if not parts:
        raise ManifestValidationError("entry must identify a file inside the plugin directory.")
    if ".." in parts:
        raise ManifestValidationError("entry must not contain parent-directory traversal.")

    return parts


def resolve_entry_path(plugin_dir: str | Path, entry: object) -> Path:
    """Resolve an entry path and ensure it remains inside the plugin root."""
    root = Path(plugin_dir).resolve(strict=False)
    candidate = root.joinpath(*_entry_parts(entry)).resolve(strict=False)

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestValidationError(
            "entry resolves outside the plugin directory."
        ) from exc

    return candidate


def validate_manifest(
    manifest: object,
    *,
    plugin_dir: str | Path | None = None,
) -> dict:
    """Validate manifest syntax and return an unmodified shallow copy."""
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a JSON object.")

    validated = dict(manifest)

    for field_name in REQUIRED_MANIFEST_FIELDS:
        value = validated.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ManifestValidationError(
                f"{field_name} is required and must be a non-empty string."
            )

    plugin_id = validated["id"]
    if PLUGIN_ID_PATTERN.fullmatch(plugin_id) is None:
        raise ManifestValidationError(
            "id must match ^[a-z][a-z0-9_]*$."
        )

    try:
        parse_version(validated["version"], field_name="version")
    except PluginVersionError as exc:
        raise ManifestValidationError(str(exc)) from exc

    _entry_parts(validated["entry"])
    if plugin_dir is not None:
        resolve_entry_path(plugin_dir, validated["entry"])

    if "schema_version" in validated:
        schema_version = validated["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ManifestValidationError("schema_version must be the integer 1.")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ManifestValidationError(
                f"Unsupported schema_version: {schema_version}."
            )

    minimum = validated.get("min_app_version")
    maximum = validated.get("max_app_version")
    try:
        parsed_minimum = (
            parse_version(minimum, field_name="min_app_version")
            if "min_app_version" in validated
            else None
        )
        parsed_maximum = (
            parse_version(maximum, field_name="max_app_version")
            if "max_app_version" in validated
            else None
        )
    except PluginVersionError as exc:
        raise ManifestValidationError(str(exc)) from exc

    if (
        parsed_minimum is not None
        and parsed_maximum is not None
        and parsed_minimum > parsed_maximum
    ):
        raise ManifestValidationError(
            "min_app_version must not be greater than max_app_version."
        )

    return validated


def check_manifest_compatibility(
    manifest: object,
    app_version: object,
) -> CompatibilityResult:
    """Validate a manifest and check its declared application-version bounds."""
    validated = validate_manifest(manifest)
    return check_app_compatibility(
        app_version,
        min_app_version=validated.get("min_app_version"),
        max_app_version=validated.get("max_app_version"),
    )
