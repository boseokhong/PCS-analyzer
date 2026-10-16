from __future__ import annotations

import json
import http.client
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from app_version import APP_VERSION
from plugin_system.plugin_manifest import PLUGIN_ID_PATTERN
from plugin_system.plugin_version import (
    PluginVersionError,
    check_app_compatibility,
    compare_versions,
    parse_version,
)


CATALOG_SCHEMA_VERSION = 1
MAX_CATALOG_RESPONSE_SIZE = 5 * 1024 * 1024
SUPPORTED_STATUSES = frozenset({"active", "deprecated", "unavailable", "blocked"})
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class PluginRepositoryError(Exception):
    code = "repository_error"

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        plugin_id: str | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.plugin_id = plugin_id
        self.index = index


class RepositoryURLValidationError(PluginRepositoryError):
    code = "invalid_url"


class RepositoryNetworkError(PluginRepositoryError):
    code = "network_error"


class RepositoryTimeoutError(RepositoryNetworkError):
    code = "timeout"


class CatalogResponseTooLargeError(PluginRepositoryError):
    code = "response_too_large"


class CatalogJSONError(PluginRepositoryError):
    code = "invalid_json"


class CatalogSchemaError(PluginRepositoryError):
    code = "invalid_catalog_schema"


class PluginMetadataError(CatalogSchemaError):
    code = "invalid_plugin_metadata"


@dataclass(frozen=True)
class CatalogPlugin:
    id: str
    name: str
    latest_version: str
    author: str
    description: str
    homepage_url: str | None
    source_url: str | None
    download_url: str | None
    sha256: str | None
    min_app_version: str | None
    max_app_version: str | None
    status: str
    category: str | None
    tags: tuple[str, ...]
    license: str | None


@dataclass(frozen=True)
class PluginCatalog:
    schema_version: int
    plugins: tuple[CatalogPlugin, ...]


@dataclass(frozen=True)
class RepositoryErrorInfo:
    code: str
    message: str
    url: str | None
    plugin_id: str | None = None
    index: int | None = None


@dataclass(frozen=True)
class RepositoryFetchResult:
    ok: bool
    catalog: PluginCatalog | None = None
    error: RepositoryErrorInfo | None = None


@dataclass(frozen=True)
class ClassifiedPlugin:
    remote: CatalogPlugin
    local_record: Mapping[str, Any] | None
    reason: str | None = None


@dataclass(frozen=True)
class LocalOnlyPlugin:
    plugin_id: str
    local_record: Mapping[str, Any]


@dataclass(frozen=True)
class CatalogClassification:
    available: tuple[ClassifiedPlugin, ...]
    installed: tuple[ClassifiedPlugin, ...]
    updates: tuple[ClassifiedPlugin, ...]
    incompatible: tuple[ClassifiedPlugin, ...]
    deprecated_unavailable: tuple[ClassifiedPlugin, ...]
    local_only: tuple[LocalOnlyPlugin, ...]
    local_errors: tuple[ClassifiedPlugin, ...]


def _validate_http_url(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    value = value.strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"{field_name} must be an absolute HTTP(S) URL with a host."
        )
    return value


def _optional_string(entry: Mapping, field_name: str) -> str | None:
    value = entry.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when present.")
    return value.strip()


def _required_string(entry: Mapping, field_name: str) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and must be a non-empty string.")
    return value.strip()


def _parse_plugin_entry(raw_entry: object, index: int) -> CatalogPlugin:
    if not isinstance(raw_entry, Mapping):
        raise PluginMetadataError(
            f"Plugin entry at index {index} must be a JSON object.", index=index
        )

    plugin_id = raw_entry.get("id") if isinstance(raw_entry.get("id"), str) else None
    try:
        plugin_id = _required_string(raw_entry, "id")
        if PLUGIN_ID_PATTERN.fullmatch(plugin_id) is None:
            raise ValueError("id must match ^[a-z][a-z0-9_]*$.")

        name = _required_string(raw_entry, "name")
        latest_version = _required_string(raw_entry, "latest_version")
        parse_version(latest_version, field_name="latest_version")
        author = _required_string(raw_entry, "author")
        description = _required_string(raw_entry, "description")

        homepage_url = _optional_string(raw_entry, "homepage_url")
        source_url = _optional_string(raw_entry, "source_url")
        if homepage_url is None and source_url is None:
            raise ValueError("At least one of homepage_url or source_url is required.")
        if homepage_url is not None:
            homepage_url = _validate_http_url(homepage_url, "homepage_url")
        if source_url is not None:
            source_url = _validate_http_url(source_url, "source_url")

        status = _required_string(raw_entry, "status")
        if status not in SUPPORTED_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(SUPPORTED_STATUSES)}."
            )

        download_url = _optional_string(raw_entry, "download_url")
        sha256 = _optional_string(raw_entry, "sha256")
        if status == "active" and (download_url is None or sha256 is None):
            raise ValueError("Active plugins require download_url and sha256.")
        if (download_url is None) != (sha256 is None):
            raise ValueError("download_url and sha256 must be supplied together.")
        if download_url is not None:
            download_url = _validate_http_url(download_url, "download_url")
            if SHA256_PATTERN.fullmatch(sha256 or "") is None:
                raise ValueError("sha256 must contain exactly 64 hexadecimal characters.")
            sha256 = (sha256 or "").lower()

        min_app_version = _optional_string(raw_entry, "min_app_version")
        max_app_version = _optional_string(raw_entry, "max_app_version")
        parsed_minimum = (
            parse_version(min_app_version, field_name="min_app_version")
            if min_app_version is not None
            else None
        )
        parsed_maximum = (
            parse_version(max_app_version, field_name="max_app_version")
            if max_app_version is not None
            else None
        )
        if (
            parsed_minimum is not None
            and parsed_maximum is not None
            and parsed_minimum > parsed_maximum
        ):
            raise ValueError(
                "min_app_version must not be greater than max_app_version."
            )

        category = _optional_string(raw_entry, "category")
        license_name = _optional_string(raw_entry, "license")
        raw_tags = raw_entry.get("tags", [])
        if not isinstance(raw_tags, list):
            raise ValueError("tags must be a list of non-empty strings.")
        tags = []
        seen_tags = set()
        for tag in raw_tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("tags must be a list of non-empty strings.")
            normalized_tag = tag.strip()
            tag_key = normalized_tag.casefold()
            if tag_key in seen_tags:
                raise ValueError(f"Duplicate tag: {normalized_tag!r}.")
            seen_tags.add(tag_key)
            tags.append(normalized_tag)

        return CatalogPlugin(
            id=plugin_id,
            name=name,
            latest_version=latest_version,
            author=author,
            description=description,
            homepage_url=homepage_url,
            source_url=source_url,
            download_url=download_url,
            sha256=sha256,
            min_app_version=min_app_version,
            max_app_version=max_app_version,
            status=status,
            category=category,
            tags=tuple(tags),
            license=license_name,
        )
    except (ValueError, PluginVersionError) as exc:
        raise PluginMetadataError(
            f"Invalid plugin metadata at index {index}: {exc}",
            plugin_id=plugin_id,
            index=index,
        ) from exc


def parse_catalog(data: object) -> PluginCatalog:
    if not isinstance(data, Mapping):
        raise CatalogSchemaError("Catalog must be a JSON object.")

    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise CatalogSchemaError("schema_version must be the integer 1.")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogSchemaError(
            f"Unsupported catalog schema_version: {schema_version}."
        )

    raw_plugins = data.get("plugins")
    if not isinstance(raw_plugins, list):
        raise CatalogSchemaError("plugins must be a JSON array.")

    plugins = []
    seen_ids = set()
    for index, raw_entry in enumerate(raw_plugins):
        plugin = _parse_plugin_entry(raw_entry, index)
        if plugin.id in seen_ids:
            raise PluginMetadataError(
                f"Duplicate plugin id {plugin.id!r} at index {index}.",
                plugin_id=plugin.id,
                index=index,
            )
        seen_ids.add(plugin.id)
        plugins.append(plugin)

    return PluginCatalog(
        schema_version=schema_version,
        plugins=tuple(plugins),
    )


def parse_catalog_json(text_or_bytes: str | bytes) -> PluginCatalog:
    if isinstance(text_or_bytes, bytes):
        try:
            text = text_or_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CatalogJSONError(f"Catalog is not valid UTF-8: {exc}") from exc
    elif isinstance(text_or_bytes, str):
        text = text_or_bytes
    else:
        raise CatalogJSONError("Catalog response must be text or bytes.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogJSONError(f"Catalog contains invalid JSON: {exc}") from exc
    return parse_catalog(data)


def _read_limited_response(response: object) -> bytes:
    headers = getattr(response, "headers", None)
    if headers is not None:
        content_length = headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                declared_size = None
            if declared_size is not None and declared_size > MAX_CATALOG_RESPONSE_SIZE:
                raise CatalogResponseTooLargeError(
                    f"Catalog response exceeds the {MAX_CATALOG_RESPONSE_SIZE}-byte limit."
                )

    chunks = []
    total = 0
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_CATALOG_RESPONSE_SIZE:
            raise CatalogResponseTooLargeError(
                f"Catalog response exceeds the {MAX_CATALOG_RESPONSE_SIZE}-byte limit."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_catalog(
    url: str,
    *,
    timeout: float = 10.0,
    opener: Any = None,
) -> PluginCatalog:
    try:
        validated_url = _validate_http_url(url, "catalog URL")
    except ValueError as exc:
        raise RepositoryURLValidationError(str(exc), url=str(url)) from exc

    request = urllib.request.Request(
        validated_url,
        headers={"User-Agent": "PCS-Analyzer-Plugin-Repository/1"},
    )
    open_url = opener or urllib.request.urlopen

    try:
        response = open_url(request, timeout=timeout)
        try:
            payload = _read_limited_response(response)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    except CatalogResponseTooLargeError as exc:
        if exc.url is None:
            exc.url = validated_url
        raise
    except (socket.timeout, TimeoutError) as exc:
        raise RepositoryTimeoutError(
            f"Timed out while fetching plugin catalog: {exc}", url=validated_url
        ) from exc
    except urllib.error.HTTPError as exc:
        raise RepositoryNetworkError(
            f"Plugin catalog request failed with HTTP {exc.code}: {exc.reason}",
            url=validated_url,
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise RepositoryTimeoutError(
                f"Timed out while fetching plugin catalog: {exc.reason}",
                url=validated_url,
            ) from exc
        raise RepositoryNetworkError(
            f"Could not fetch plugin catalog: {exc.reason}", url=validated_url
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise RepositoryNetworkError(
            f"Could not fetch plugin catalog: {exc}", url=validated_url
        ) from exc

    try:
        return parse_catalog_json(payload)
    except PluginRepositoryError as exc:
        if exc.url is None:
            exc.url = validated_url
        raise


def fetch_catalog_result(
    url: str,
    *,
    timeout: float = 10.0,
    opener: Any = None,
) -> RepositoryFetchResult:
    try:
        return RepositoryFetchResult(
            ok=True,
            catalog=fetch_catalog(url, timeout=timeout, opener=opener),
        )
    except PluginRepositoryError as exc:
        return RepositoryFetchResult(
            ok=False,
            error=RepositoryErrorInfo(
                code=exc.code,
                message=str(exc),
                url=exc.url,
                plugin_id=exc.plugin_id,
                index=exc.index,
            ),
        )


def _snapshot_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(record))


def classify_catalog(
    catalog: PluginCatalog,
    local_registry: Mapping[str, Any],
    *,
    app_version: object = APP_VERSION,
) -> CatalogClassification:
    if not isinstance(catalog, PluginCatalog):
        raise CatalogSchemaError("catalog must be a PluginCatalog instance.")
    if not isinstance(local_registry, Mapping):
        raise CatalogSchemaError("local_registry must be a mapping.")

    groups: dict[str, list] = {
        "available": [],
        "installed": [],
        "updates": [],
        "incompatible": [],
        "deprecated_unavailable": [],
        "local_errors": [],
    }
    remote_ids = {plugin.id for plugin in catalog.plugins}

    for remote in sorted(catalog.plugins, key=lambda plugin: plugin.id):
        has_local = remote.id in local_registry
        raw_local = local_registry.get(remote.id)
        local_snapshot = None
        local_version = None

        if has_local:
            if not isinstance(raw_local, Mapping):
                groups["local_errors"].append(
                    ClassifiedPlugin(
                        remote,
                        None,
                        "local registry record is not an object",
                    )
                )
                continue
            local_snapshot = _snapshot_record(raw_local)
            raw_version = raw_local.get("version")
            if not isinstance(raw_version, str) or not raw_version.strip():
                groups["local_errors"].append(
                    ClassifiedPlugin(remote, local_snapshot, "missing local version")
                )
                continue
            try:
                parse_version(raw_version, field_name="local version")
            except PluginVersionError as exc:
                groups["local_errors"].append(
                    ClassifiedPlugin(
                        remote,
                        local_snapshot,
                        f"invalid local version: {exc}",
                    )
                )
                continue
            local_version = raw_version

        if remote.status != "active":
            reason = f"catalog status is {remote.status}"
            if remote.download_url is None:
                reason += "; package is not distributable"
            groups["deprecated_unavailable"].append(
                ClassifiedPlugin(remote, local_snapshot, reason)
            )
            continue

        try:
            compatibility = check_app_compatibility(
                app_version,
                min_app_version=remote.min_app_version,
                max_app_version=remote.max_app_version,
            )
        except PluginVersionError as exc:
            raise CatalogSchemaError(
                f"Could not classify catalog against app version: {exc}"
            ) from exc
        if not compatibility.compatible:
            groups["incompatible"].append(
                ClassifiedPlugin(remote, local_snapshot, compatibility.reason)
            )
            continue

        if not has_local:
            groups["available"].append(ClassifiedPlugin(remote, None))
            continue

        relation = compare_versions(remote.latest_version, local_version)
        if relation > 0:
            groups["updates"].append(ClassifiedPlugin(remote, local_snapshot))
        elif relation < 0:
            groups["installed"].append(
                ClassifiedPlugin(
                    remote,
                    local_snapshot,
                    "local version is newer than catalog version",
                )
            )
        else:
            groups["installed"].append(ClassifiedPlugin(remote, local_snapshot))

    local_only = []
    for plugin_id in sorted(set(local_registry) - remote_ids, key=str):
        record = local_registry[plugin_id]
        if not isinstance(plugin_id, str) or not isinstance(record, Mapping):
            raise CatalogSchemaError(
                "Local-only registry entries must use string ids and object records."
            )
        local_only.append(
            LocalOnlyPlugin(plugin_id, _snapshot_record(record))
        )

    return CatalogClassification(
        available=tuple(groups["available"]),
        installed=tuple(groups["installed"]),
        updates=tuple(groups["updates"]),
        incompatible=tuple(groups["incompatible"]),
        deprecated_unavailable=tuple(groups["deprecated_unavailable"]),
        local_only=tuple(local_only),
        local_errors=tuple(groups["local_errors"]),
    )
