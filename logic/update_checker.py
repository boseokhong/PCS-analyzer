# logic/update_checker.py

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class UpdateCheckResult:
    current_version: str
    latest_version: Optional[str]
    latest_version_raw: Optional[str]
    status: str
    release_url: str
    error: Optional[str] = None


def normalize_version_text(version: str) -> str:
    """
    Display-normalized version text.
    'v1.3.3' -> '1.3.3'
    '1.3.3'  -> '1.3.3'
    """
    s = str(version).strip()
    if s.lower().startswith("v"):
        s = s[1:]
    return s


def version_tuple(version: str) -> tuple[int, int, int]:
    """
    Convert 'v1.3.3', '1.3.3', or '1.3.3-beta'
    into a comparable tuple.
    """
    s = normalize_version_text(version).lower()
    nums = re.findall(r"\d+", s)
    nums = [int(x) for x in nums[:3]]

    while len(nums) < 3:
        nums.append(0)

    return tuple(nums)


def check_latest_release(
    current_version: str,
    api_url: str,
    fallback_release_url: str,
    timeout: float = 12.0,
) -> UpdateCheckResult:
    try:
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "PCS-Analyzer-Update-Checker",
            },
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))

        latest_raw = str(data.get("tag_name", "")).strip()
        release_url = data.get("html_url") or fallback_release_url

        if not latest_raw:
            return UpdateCheckResult(
                current_version=normalize_version_text(current_version),
                latest_version=None,
                latest_version_raw=None,
                status="error",
                release_url=fallback_release_url,
                error="Could not read the latest release tag.",
            )

        current_display = normalize_version_text(current_version)
        latest_display = normalize_version_text(latest_raw)

        current_v = version_tuple(current_version)
        latest_v = version_tuple(latest_raw)

        if latest_v > current_v:
            status = "update_available"
        elif latest_v == current_v:
            status = "up_to_date"
        else:
            status = "ahead_of_release"

        return UpdateCheckResult(
            current_version=current_display,
            latest_version=latest_display,
            latest_version_raw=latest_raw,
            status=status,
            release_url=release_url,
            error=None,
        )

    except Exception as exc:
        return UpdateCheckResult(
            current_version=normalize_version_text(current_version),
            latest_version=None,
            latest_version_raw=None,
            status="error",
            release_url=fallback_release_url,
            error=str(exc),
        )