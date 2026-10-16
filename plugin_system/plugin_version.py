from __future__ import annotations

from dataclasses import dataclass

from packaging.version import InvalidVersion, Version


class PluginVersionError(ValueError):
    """Raised when plugin or application version metadata is invalid."""


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason: str | None = None


def parse_version(value: object, *, field_name: str = "version") -> Version:
    """Parse a version field and provide a field-specific validation error."""
    if not isinstance(value, str):
        raise PluginVersionError(f"{field_name} must be a non-empty string.")

    value = value.strip()
    if not value:
        raise PluginVersionError(f"{field_name} must be a non-empty string.")

    try:
        return Version(value)
    except InvalidVersion as exc:
        raise PluginVersionError(
            f"Invalid {field_name} value {value!r}: {exc}"
        ) from exc


def compare_versions(left: object, right: object) -> int:
    """Return -1, 0, or 1 according to PEP 440 version ordering."""
    left_version = parse_version(left, field_name="left version")
    right_version = parse_version(right, field_name="right version")

    if left_version < right_version:
        return -1
    if left_version > right_version:
        return 1
    return 0


def check_app_compatibility(
    app_version: object,
    *,
    min_app_version: object | None = None,
    max_app_version: object | None = None,
) -> CompatibilityResult:
    """Check inclusive application-version bounds for a plugin."""
    current = parse_version(app_version, field_name="app_version")
    minimum = (
        parse_version(min_app_version, field_name="min_app_version")
        if min_app_version is not None
        else None
    )
    maximum = (
        parse_version(max_app_version, field_name="max_app_version")
        if max_app_version is not None
        else None
    )

    if minimum is not None and maximum is not None and minimum > maximum:
        raise PluginVersionError(
            "min_app_version must not be greater than max_app_version."
        )

    if minimum is not None and current < minimum:
        return CompatibilityResult(
            compatible=False,
            reason=(
                f"PCS Analyzer {current} is older than the required minimum "
                f"version {minimum}."
            ),
        )

    if maximum is not None and current > maximum:
        return CompatibilityResult(
            compatible=False,
            reason=(
                f"PCS Analyzer {current} is newer than the supported maximum "
                f"version {maximum}."
            ),
        )

    return CompatibilityResult(compatible=True)
