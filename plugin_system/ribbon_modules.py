"""Presentation metadata and application-global ordering for module launchers."""
from plugin_system.tool_registry import ToolLauncher


def category_icon(category):
    value = str(category or "").strip().casefold()
    if value in {"nmr / pcs", "nmr/pcs", "nmr", "pcs", "pcs / dynamics"}:
        return "nmr_workbench"
    if value in {"structure", "structure analysis"}:
        return "chemical_map"
    if value in {"visualization", "visualisation"}:
        return "eye"
    return "toolbox"


def normalize_order(value):
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item.strip()))


def ordered_launchers(launchers, order):
    by_id = {tool.id: tool for tool in launchers}
    saved = normalize_order(order)
    first = [by_id.pop(key) for key in saved if key in by_id]
    return first + sorted(by_id.values(), key=lambda tool: (tool.label.casefold(), tool.id))


def loaded_launchers(state):
    loaded = {result.get("plugin_id") for result in state.get("plugin_load_results", ())
              if result.get("status") == "loaded"}
    output = []
    for tool in state.get("_module_ribbon_registrations", ()):
        if tool.source not in loaded:
            continue
        # Legacy modules sometimes register the same command through both APIs.
        duplicate = next((index for index, peer in enumerate(output)
                          if peer.source == tool.source and peer.launch == tool.launch), None)
        if duplicate is not None:
            if tool.id.startswith(f"{tool.source}:tool:"):
                output[duplicate] = tool
            continue
        output.append(tool)
    return output


def capture_launcher(app, label, command, *, tool_id=None, category=None, enabled=None, tooltip=""):
    source = getattr(app, "current_plugin_id", None)
    if not source:
        return
    manifest = getattr(app, "current_plugin_manifest", None) or {}
    tools = app.state.setdefault("_module_ribbon_registrations", [])
    # Label-based identity survives unrelated insertions in a multi-command module.
    base = f"{source}:tool:{tool_id}" if tool_id else f"{source}:menu:{label}"
    if tool_id:
        tools[:] = [tool for tool in tools if tool.id != base]
    identifier = base
    suffix = 2
    while any(tool.id == identifier for tool in tools):
        identifier = f"{base}:{suffix}"
        suffix += 1
    tools.append(ToolLauncher(identifier, str(label),
                              str(category or manifest.get("category") or "Other / Uncategorized"),
                              command, source=str(source), enabled=enabled, tooltip=tooltip))
