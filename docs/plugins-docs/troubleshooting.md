# Plugin Troubleshooting

## Installed but absent from the Modules menu

Open **Module Manager → Installed**. Check **Enabled**, **Session status** and
**Details**. Installed means files/metadata are present, not that registration
succeeded. Details can identify missing entry files, dependencies, incompatible
versions, import failures or `register(app)` errors, with technical context.
A plugin may register a Quick tools launcher through `register_tool()` rather
than only adding a direct menu entry.

Enable a disabled plugin, then inspect the new session status. Successful local
file/folder/ZIP installations now refresh the Modules menu. For a replacement,
restart first: a menu refresh may reuse old imported code.

## Catalog or download fails

Read **Last operation details** for the network, metadata, compatibility,
package-ID/version or SHA-256 error. Check the internet connection and retry
**Refresh Catalog** when appropriate. A failed fetch does not remove local
modules; a previously loaded catalog may remain visible. Do not bypass a hash
mismatch by assuming the package is safe. Check the release with its maintainer.

## Incompatible, blocked or local-only

Use **Details** to read the reason. Incompatible bounds prevent new installation
and valid declared bounds also affect loading. Blocked/unavailable/deprecated
catalog entries are not offered as normal install/update actions. Local-only
means the installed ID is absent from that catalog; it is not a load diagnosis.
Legacy compatibility warnings deserve review even when Session status is Loaded.

## Refresh does not activate an updated version

Installed **Refresh** rereads local records/diagnostics; **Refresh Catalog**
retrieves catalog metadata. Neither unloads Python modules. After an update or
replacement, close plugin windows and restart PCS Analyzer to execute the new
files. Disabling/removing also does not reverse arbitrary runtime side effects.

## Replacement or removal fails on Windows

Close plugin windows and restart before retrying. File locks or permissions can
prevent replacement, rollback or cleanup. **Last operation details** reports
restart requirements and recovery paths. Preserve those paths and the message;
a failed operation is not a guarantee that rollback completed. Avoid deleting
an installation or backup manually until the application is closed and the
reported state is understood.

## Registry cannot be read

Malformed registry JSON or invalid records are reported as errors rather than
silently treated as an empty installed list. Preserve the registry file before
repair and provide its error details to the maintainer. Do not reinstall all
modules merely because a damaged registry cannot be displayed.

## Missing dependency

Manifest dependency lists are not installed automatically. Follow the plugin's
instructions for the Python environment running PCS Analyzer. Frozen builds
may need a plugin-specific distribution strategy. Report the missing package
from Installed **Details** to the plugin author.

## Author checks

- **Tool opens during startup:** defer launch with `command=lambda: open_plugin(app)`;
  keep standalone execution behind `if __name__ == "__main__"`.
- **Relative import fails:** test the installed package layout, including sibling
  modules, rather than executing the entry file directly.
- **Resource missing:** resolve resource paths relative to `Path(__file__).resolve().parent`,
  not the process working directory.
- **One plugin fails while others load:** load-error isolation is expected, but
  import-time state changes are not a sandbox or rollback boundary. Restart
  before diagnosis; later tool errors remain the plugin's responsibility.

See [Creating Plugins](creating-plugins.md), [Manifest](manifest.md) and the
[API reference](plugin-api.md) for author-specific details.
