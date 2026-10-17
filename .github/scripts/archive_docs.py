"""Migrate an already built legacy site to mike without rebuilding its content.

Run against a disposable checkout of the publication branch, never the source
checkout. Git commits and pushes are deliberately left to the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil


def redirect(target: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>Redirecting</title><script>location.replace('
        + json.dumps(target)
        + '+location.search+location.hash);</script>'
        '<noscript><meta http-equiv="refresh" content="0;url='
        + html.escape(target, quote=True)
        + '"></noscript></head><body><a href="'
        + html.escape(target, quote=True)
        + '">Open documentation</a></body></html>\n'
    )


def archive_site(site: Path, version: str, prefix: str, source_commit: str) -> dict:
    site = site.resolve()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("The archive version must be MAJOR.MINOR.PATCH")
    if not re.fullmatch(r"/[A-Za-z0-9_/-]*/", prefix) or "//" in prefix:
        raise ValueError("Use an absolute site path with a trailing slash")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Record the full source publication commit")
    if (site / "mkdocs.yml").exists():
        raise ValueError("Refusing to migrate a source checkout")
    registry = site / "versions.json"
    if registry.exists():
        records = json.loads(registry.read_text(encoding="utf-8"))
        if not any(item.get("version") == version for item in records):
            raise ValueError("A versioned site exists without the expected legacy archive")
        manifest_path = site / version / "archive-provenance.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_commit"] != source_commit:
            raise ValueError("Existing archive has a different source publication commit")
        for name, digest in manifest["assets_sha256"].items():
            if hashlib.sha256((site / version / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Archived asset changed: " + name)
        return {"status": "already_archived", "version": version}
    destination = site / version
    if destination.exists():
        raise ValueError("Archive destination already exists")
    for required in ("index.html", "assets", "images", "search/search_index.json"):
        if not (site / required).exists():
            raise ValueError("Missing legacy site resource: " + required)
    files = sorted(p for p in site.rglob("*") if ".git" not in p.relative_to(site).parts and p.is_file())
    if any(p.is_symlink() for p in files):
        raise ValueError("Legacy publication must contain regular files")
    hashes = {p.relative_to(site).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    # Snapshot all resources before adding version files or replacing root routes.
    for source in files:
        target = destination / source.relative_to(site)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    version_prefix = prefix + version + "/"
    changed_html = []
    for source in files:
        relative = source.relative_to(site)
        if source.suffix != ".html":
            continue
        target = destination / relative
        content = target.read_bytes().decode("utf-8")

        def configure(match):
            config = json.loads(match.group(2))
            config["version"] = {"provider": "mike", "default": "latest"}
            if relative.as_posix() == "404.html":
                config["base"] = version_prefix.rstrip("/")
                if config.get("search", "").startswith("/"):
                    config["search"] = version_prefix + config["search"].lstrip("/")
            return match.group(1) + json.dumps(config, ensure_ascii=False) + match.group(3)

        content, count = re.subn(
            r'(<script id="__config" type="application/json">)(.*?)(</script>)',
            configure, content, flags=re.S,
        )
        if count != 1:
            raise ValueError("Expected one Material configuration in " + str(relative))
        # Legacy 404 navigation/assets were rooted at /, not the project prefix.
        content = re.sub(r'((?:href|src)=")/(?!/)', lambda m: m.group(1) + version_prefix, content)
        content = content.replace('new URL("/",location)', 'new URL(' + json.dumps(version_prefix) + ',location)')
        target.write_bytes(content.encode("utf-8"))
        changed_html.append(relative.as_posix())
        if relative.as_posix() != "404.html":
            route = relative.as_posix().removesuffix("index.html")
            source.write_text(redirect(prefix + "latest/" + route), encoding="utf-8")
            alias = site / "latest" / relative
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.write_text(redirect(version_prefix + route), encoding="utf-8")
    error = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Page not found</title><h1>404 — Page not found</h1><p><a href="' + html.escape(prefix + 'latest/', quote=True) + '">Open the latest documentation</a></p></html>\n'
    (site / "404.html").write_text(error, encoding="utf-8")
    assets = {name: digest for name, digest in hashes.items() if not name.endswith(".html")}
    for name, digest in assets.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Asset copy mismatch: " + name)
    provenance = {
        "version": version, "source_commit": source_commit,
        "original_sha256": hashes, "assets_sha256": assets,
        "adapted_html": changed_html,
        "adaptations": "Material version-selector configuration and version-relative root links only; page content was not rebuilt",
    }
    (destination / "archive-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    registry.write_text(json.dumps([{"version": version, "title": version, "aliases": ["latest"]}], indent=2) + "\n", encoding="utf-8")
    return {"status": "archived", "version": version, "files": len(files), "unchanged_assets": len(assets)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--version", default="1.3.4")
    parser.add_argument("--site-prefix", default="/PCS-analyzer/")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(archive_site(args.site_dir, args.version, args.site_prefix, args.source_commit)))
