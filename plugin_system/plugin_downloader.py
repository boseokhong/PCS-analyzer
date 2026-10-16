from __future__ import annotations

import hashlib
import hmac
import http.client
import os
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from plugin_system import plugin_installer
from plugin_system.plugin_repository import CatalogPlugin


DEFAULT_MAX_PACKAGE_SIZE = 512 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class PluginDownloadError(Exception):
    stage = "download"

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        cleanup_error: str | None = None,
        cleanup_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.cleanup_error = cleanup_error
        self.cleanup_path = cleanup_path


class PackageMetadataError(PluginDownloadError):
    stage = "metadata"


class PackageDownloadError(PluginDownloadError):
    stage = "download"


class PackageDownloadTimeoutError(PackageDownloadError):
    pass


class PackageTooLargeError(PackageDownloadError):
    def __init__(
        self,
        message: str,
        *,
        limit: int,
        actual_size: int | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message, url=url)
        self.limit = limit
        self.actual_size = actual_size


class PackageHashMismatchError(PluginDownloadError):
    stage = "verification"

    def __init__(
        self,
        message: str,
        *,
        expected_hash: str,
        actual_hash: str,
        url: str | None = None,
    ) -> None:
        super().__init__(message, url=url)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


def _validate_http_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackageMetadataError("Plugin package download_url is missing.")
    value = value.strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PackageMetadataError(
            "Plugin package download_url must be an absolute HTTP(S) URL with a host.",
            url=value,
        )
    return value


def _validate_plugin_metadata(
    plugin: CatalogPlugin,
    max_size: object,
) -> tuple[str, str, int]:
    if not isinstance(plugin, CatalogPlugin):
        raise PackageMetadataError("plugin must be a CatalogPlugin instance.")
    if plugin.status != "active":
        raise PackageMetadataError(
            f"Only active catalog plugins can be downloaded; status is {plugin.status!r}.",
            url=plugin.download_url,
        )
    download_url = _validate_http_url(plugin.download_url)
    if not isinstance(plugin.sha256, str) or SHA256_PATTERN.fullmatch(plugin.sha256) is None:
        raise PackageMetadataError(
            "Active plugin package requires a valid 64-character SHA-256 value.",
            url=download_url,
        )
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
        raise PackageMetadataError("max_size must be a positive integer.", url=download_url)
    return download_url, plugin.sha256.lower(), max_size


def _close_response(response: object) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Response cleanup must not replace a transport, integrity, or install result.
            pass


def _delete_temp_file(path: Path) -> str | None:
    try:
        path.unlink(missing_ok=True)
        return None
    except OSError as exc:
        return f"Temporary package file could not be deleted: {exc}"


def _attach_cleanup_failure(
    exc: PluginDownloadError,
    path: Path,
    cleanup_error: str | None,
) -> None:
    if cleanup_error is not None:
        exc.cleanup_error = cleanup_error
        exc.cleanup_path = path


def _open_package_response(url: str, timeout: float, opener: Any) -> object:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PCS-Analyzer-Plugin-Downloader/1"},
    )
    open_url = opener or urllib.request.urlopen
    try:
        return open_url(request, timeout=timeout)
    except (socket.timeout, TimeoutError) as exc:
        raise PackageDownloadTimeoutError(
            f"Timed out while downloading plugin package: {exc}", url=url
        ) from exc
    except urllib.error.HTTPError as exc:
        raise PackageDownloadError(
            f"Plugin package request failed with HTTP {exc.code}: {exc.reason}",
            url=url,
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise PackageDownloadTimeoutError(
                f"Timed out while downloading plugin package: {exc.reason}", url=url
            ) from exc
        raise PackageDownloadError(
            f"Could not download plugin package: {exc.reason}", url=url
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise PackageDownloadError(
            f"Remote plugin package transport failed: {exc}", url=url
        ) from exc


def _read_content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _read_remote_chunk(response: object, url: str) -> bytes:
    try:
        return response.read(DOWNLOAD_CHUNK_SIZE)
    except (socket.timeout, TimeoutError) as exc:
        raise PackageDownloadTimeoutError(
            f"Timed out while reading plugin package: {exc}", url=url
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise PackageDownloadTimeoutError(
                f"Timed out while reading plugin package: {exc.reason}", url=url
            ) from exc
        raise PackageDownloadError(
            f"Plugin package transport failed while reading: {exc.reason}", url=url
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        raise PackageDownloadError(
            f"Plugin package transport failed while reading: {exc}", url=url
        ) from exc


def download_plugin_package(
    plugin: CatalogPlugin,
    *,
    timeout: float = 30.0,
    max_size: int = DEFAULT_MAX_PACKAGE_SIZE,
    opener: Any = None,
) -> Path:
    """
    Download and verify a catalog plugin package.

    On success, ownership of the returned temporary file passes to the caller,
    which is responsible for deleting it. Partial files are cleaned on failure.
    """
    url, expected_hash, size_limit = _validate_plugin_metadata(plugin, max_size)
    response = _open_package_response(url, timeout, opener)
    temp_path = None

    try:
        declared_size = _read_content_length(response)
        if declared_size is not None and declared_size > size_limit:
            raise PackageTooLargeError(
                f"Plugin package Content-Length {declared_size} exceeds the "
                f"{size_limit}-byte limit.",
                limit=size_limit,
                actual_size=declared_size,
                url=url,
            )

        try:
            file_descriptor, temp_name = tempfile.mkstemp(
                prefix="pcs-plugin-download-", suffix=".zip"
            )
            temp_path = Path(temp_name)
        except OSError as exc:
            raise PackageDownloadError(
                f"Could not create local temporary package file: {exc}", url=url
            ) from exc

        digest = hashlib.sha256()
        streamed_size = 0
        try:
            try:
                temp_file = os.fdopen(file_descriptor, "wb")
            except (OSError, ValueError) as exc:
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
                raise PackageDownloadError(
                    f"Could not open local temporary package file: {exc}", url=url
                ) from exc

            with temp_file:
                while True:
                    chunk = _read_remote_chunk(response, url)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise PackageDownloadError(
                            "Remote plugin package stream returned non-byte data.",
                            url=url,
                        )
                    streamed_size += len(chunk)
                    if streamed_size > size_limit:
                        raise PackageTooLargeError(
                            f"Streamed plugin package exceeds the {size_limit}-byte limit.",
                            limit=size_limit,
                            actual_size=streamed_size,
                            url=url,
                        )
                    try:
                        temp_file.write(chunk)
                    except OSError as exc:
                        raise PackageDownloadError(
                            f"Could not write local temporary package file: {exc}",
                            url=url,
                        ) from exc
                    digest.update(chunk)
                try:
                    temp_file.flush()
                except OSError as exc:
                    raise PackageDownloadError(
                        f"Could not finalize local temporary package file: {exc}",
                        url=url,
                    ) from exc
        except PluginDownloadError:
            raise
        except OSError as exc:
            raise PackageDownloadError(
                f"Local temporary package file I/O failed: {exc}", url=url
            ) from exc

        actual_hash = digest.hexdigest()
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise PackageHashMismatchError(
                "Downloaded plugin package SHA-256 does not match catalog metadata.",
                expected_hash=expected_hash,
                actual_hash=actual_hash,
                url=url,
            )
        return temp_path
    except PluginDownloadError as exc:
        if temp_path is not None:
            _attach_cleanup_failure(exc, temp_path, _delete_temp_file(temp_path))
        raise
    except Exception as exc:
        wrapped = PackageDownloadError(
            f"Unexpected plugin package download failure: {exc}", url=url
        )
        if temp_path is not None:
            _attach_cleanup_failure(
                wrapped, temp_path, _delete_temp_file(temp_path)
            )
        raise wrapped from exc
    finally:
        _close_response(response)


def _download_error_result(exc: PluginDownloadError) -> dict:
    result = {
        "ok": False,
        "stage": exc.stage,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "url": exc.url,
    }
    if exc.cleanup_error is not None:
        result["cleanup_warning"] = exc.cleanup_error
        result["recovery_paths"] = [str(exc.cleanup_path)] if exc.cleanup_path else []
    return result


def _augment_cleanup_warning(result: dict, path: Path, cleanup_error: str) -> dict:
    augmented = dict(result)
    warnings = list(augmented.get("warnings") or [])
    warnings.append(cleanup_error)
    augmented["warnings"] = warnings
    recovery_paths = list(augmented.get("recovery_paths") or [])
    recovery_paths.append(str(path))
    augmented["recovery_paths"] = recovery_paths
    augmented["temporary_file_cleanup_failed"] = True
    return augmented


def install_catalog_plugin(
    plugin: CatalogPlugin,
    *,
    timeout: float = 30.0,
    max_size: int = DEFAULT_MAX_PACKAGE_SIZE,
    opener: Any = None,
) -> dict:
    """Download, verify, install, and internally clean a catalog package."""
    try:
        package_path = download_plugin_package(
            plugin,
            timeout=timeout,
            max_size=max_size,
            opener=opener,
        )
    except PluginDownloadError as exc:
        return _download_error_result(exc)

    installer_result = None
    installer_exception = None
    try:
        installer_result = plugin_installer.install_plugin_zip(
            package_path,
            expected_plugin_id=plugin.id,
            expected_version=plugin.latest_version,
        )
    except Exception as exc:
        installer_exception = exc

    cleanup_error = _delete_temp_file(package_path)
    if installer_exception is not None:
        if cleanup_error is not None:
            try:
                installer_exception.cleanup_error = cleanup_error
                installer_exception.cleanup_path = package_path
            except Exception:
                pass
        raise installer_exception

    if cleanup_error is not None:
        return _augment_cleanup_warning(
            installer_result, package_path, cleanup_error
        )
    return installer_result
