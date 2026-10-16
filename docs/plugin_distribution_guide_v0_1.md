# Plugin Distribution Design Notes

These notes retain the v0.1 distribution design direction for plugin authors
and maintainers. They are **not additional release guarantees**. For current
behavior, use [Module Manager](plugins-docs/overview.md),
[Installation and distribution](plugins-docs/distribution.md),
[Manifest](plugins-docs/manifest.md) and [PluginApp](plugins-docs/plugin-api.md).

## Implemented baseline

PCS Analyzer currently supports a GitHub-hosted catalog, Browse/Installed/Updates,
release downloads with SHA-256 verification, new-package manifest/version and
compatibility validation, staged local installation, backup replacement,
attempted rollback/removal and visible session/operation diagnostics.
`register(app)` and `register_tool()` are implemented. Plugins remain unsandboxed,
Python dependencies are not resolved, and PluginApp remains unversioned.
Updated files require a restart to become the executing code.

## Distribution responsibilities

The developer repository describes and maintains the plugin: source, tests,
issues, documentation, licenses and release assets. The distribution catalog
selects which release URL, digest, version and compatibility/lifecycle metadata
PCS Analyzer offers. Publishing a developer release does not automatically
update that catalog.

For a release, keep the public ID stable and align manifest version, release
tag, ZIP name and catalog version. Prefer an explicit release ZIP containing
one manifest and its declared entry; include resources and third-party notices.
Calculate its SHA-256 from the exact published bytes. Test installation and
replacement from a clean application session before requesting catalog inclusion.

Catalog inclusion and labels do not certify security or scientific validity.
The software's available categories/lifecycle states are metadata; a formal
approval, review or governance policy must be documented separately if adopted.

## Future integrated API

A stable, versioned integrated API is still proposed. It should expose explicit
units, coordinate/tensor conventions and narrow application services, with
read-only capabilities before writable operations. Existing `app.state` access
and descriptive `state_access` do not implement permission boundaries.

Do not require `api_version` for a standalone module that only uses the current
unversioned wrapper. Do not document capability permissions, API negotiation,
or guaranteed schema stability as current functionality. Future GUI migration
should preserve the same scientific behavior and user-data compatibility.

## Author checklist

Use the canonical [development checklist](plugins-docs/creating-plugins.md#development-checklist)
and [manifest field reference](plugins-docs/manifest.md#current-field-behavior).
Document supported application versions and optional dependencies, defer work
until tool launch, and report errors at the plugin UI boundary. Preserve
standalone execution separately when provided. Test source and packaged
application environments, including Windows file locks and restart behavior.
