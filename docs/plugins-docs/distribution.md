# Plugin Installation and Distribution

For routine use, follow [Module Manager](overview.md). Browse and Updates use
the configured catalog; Installed also accepts trusted local packages.
These workflows validate new manifests and application compatibility. They do
not resolve Python dependencies or sandbox code.

## Accepted local packages

| Action in Installed | Package |
| --- | --- |
| **Install .py Plugin** | A Python file defining a `PLUGIN_INFO` dictionary and callable `register(app)`. Inspection imports the file; the installed entry is copied as `plugin.py` with a generated manifest. |
| **Install Folder** | A folder containing `manifest.json` and its declared entry. A wrapper containing only one immediate package folder is also accepted. |
| **Install ZIP** | The same package layout archived as a ZIP, at the root or inside one wrapper folder. Ambiguous multi-package layouts are rejected. |

A practical package is:

```text
my_tool/
├─ manifest.json
├─ plugin.py
├─ resources/
└─ examples/
```

Entries must stay within the package. Unsafe archive paths, symlinks and
unsupported archive entries are rejected; extraction is staged before
installation. Keep IDs stable and use versions understood by PCS Analyzer
(see [Manifest](manifest.md)). New installs are enabled, while replacement
preserves an existing plugin's enabled state.

## Catalog installation

**Refresh Catalog** reads available releases and their compatibility/lifecycle
metadata. **Install** or **Update** downloads the selected package, verifies
SHA-256 and checks that its manifest ID/version matches the catalog release
before installation. A hash mismatch, download error or invalid package is
reported through **Last operation details**. No update is installed automatically.

Local folder/ZIP/file installation does not independently authenticate its
source or require a catalog hash. Catalog integrity checks are not signatures
or a security/scientific review of the code.

## Replacement and removal behavior

Installation is prepared before replacing live files. Replacement uses a backup;
on failure the installer attempts to restore the previous files and registry.
Removal likewise preserves a recoverable copy until its record change succeeds.
If rollback or cleanup fails, warnings identify remaining recovery paths.
Read those details, preserve backups and restart before recovery; do not delete
reported folders blindly.

An updated version on disk is not necessarily the code running in this session.
**Restart after updating or replacing a plugin**, including local reinstallations.
Enable/disable or Refresh is not a substitute for a clean restart. See
[Troubleshooting](troubleshooting.md).

## Author distribution checklist

- Supply the [required manifest](manifest.md), or equivalent `PLUGIN_INFO` for a
  single-file plugin; define `register(app)` in the declared entry.
- Use a stable lowercase ID and consistent release/manifest versions.
- Keep import and registration lightweight; defer tool launch to a callback.
- State required dependencies and supported PCS Analyzer versions. Dependency
  declarations are informational; handle missing packages in the plugin.
- Resolve resources relative to the plugin directory and include their licenses.
- Test clean startup, successful loading, enable/disable, replacement and removal.
- For catalog distribution, provide the release ZIP, its SHA-256 and matching
  metadata. A developer release alone does not change the catalog.

The [distribution design notes](../plugin_distribution_guide_v0_1.md) distinguish
current catalog capabilities from future integrated API and governance proposals.
