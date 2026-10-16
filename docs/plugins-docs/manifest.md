# Plugin Manifest Reference

This is **plugin-author reference**. Users should normally manage modules through
[Module Manager](overview.md), rather than editing installed metadata.
Folder/ZIP packages provide `manifest.json`; single-file installation generates
one from `PLUGIN_INFO`.

## Minimal practical manifest

```json
{
  "schema_version": 1,
  "id": "my_tool",
  "name": "My Tool",
  "version": "1.0.0",
  "entry": "plugin.py",
  "min_app_version": "1.3.4"
}
```

## Current field behavior

| Field | New-package requirement or behavior |
| --- | --- |
| `id` | Required; matches `^[a-z][a-z0-9_]*$`. It is the installation identity. New installation does not normalize arbitrary names into valid IDs. Reusing the ID replaces that plugin. |
| `name` | Required nonempty display name. |
| `version` | Required nonempty PEP 440 version; compared as a parsed version rather than plain text. Semantic-style `MAJOR.MINOR.PATCH` is recommended. |
| `entry` | Required nonempty relative path to an existing file inside the package; traversal and unsafe paths are rejected. The module must provide `register(app)`. |
| `schema_version` | Optional; when supplied must be integer `1`. |
| `min_app_version`, `max_app_version` | Optional valid version strings; inclusive application-version bounds checked at installation. Minimum must not exceed maximum. |
| `author`, `description`, `type`, `standalone`, `license` | Descriptive metadata; `type`/`standalone` do not select a different host API. |
| `dependencies`, `optional_dependencies` | Descriptive; no dependency installation or resolution. |
| `category` | Optional functional category for Ribbon icons. Use `NMR / PCS`, `Structure`, `Visualization`, or `Other / Uncategorized`. Missing/unrecognized values use the generic icon. |
| `state_access` | Descriptive intent only; no permission enforcement. |
| `api_version`, `entry_function` | No API negotiation or alternate entry function is implemented. The host calls `register(app)`. |
| `enabled` | Not the enabled-state control. Enable/Disable is application-managed in Module Manager. |

Unknown fields may remain in the supplied manifest without being acted on.
Legacy aliases such as `requires` and `optional_requires` are not dependency
resolution mechanisms. Catalog category/source links belong to catalog metadata;
do not assume every manifest field appears in Browse.

## Single-file PLUGIN_INFO

Supply a valid `id`, `name`, `version` and a callable `register(app)`.
The generated manifest uses `entry: plugin.py`. It copies author, description,
standalone and dependencies, plus supplied schema version, type, optional
dependencies, application-version bounds, license and category. Other keys are not
necessarily copied. Single-file inspection executes Python code.

## Legacy installed plugins

Loading an already installed legacy manifest is more permissive than installing
a new package. Its omitted entry can default to `plugin.py`; a manifest/installed
ID disagreement is reported as a warning. Valid application-version bounds are
checked, while malformed bounds may be ignored with diagnostics. New installs
must satisfy the stricter validation above. Inspect **Session status → Details**
after updating old packages; do not interpret legacy acceptance as API stability.

Changing a public ID creates a different installation identity. Keep IDs and
release metadata consistent, and see [Creating Plugins](creating-plugins.md)
for a working entry pattern.
