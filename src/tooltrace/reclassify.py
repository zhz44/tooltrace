"""Re-label recorded traces with the current classifier.

Every row keeps the full command in ``command_raw``, so when the classifier
changes the agent does not have to be run again: ``tool_name`` can be
recomputed from the stored command.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .classifier import classify_tool


def reclassify_file(path, backup=False):
    """Rewrite one trace file in place. Returns (rows, rows_changed)."""
    path = Path(path)
    if backup:
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))

    lines = []
    changed = 0
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            row = json.loads(line)
            new_name = classify_tool(row.get("command_raw", ""))
            if row.get("tool_name") != new_name:
                row["tool_name"] = new_name
                changed += 1
            lines.append(json.dumps(row))

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("".join(line + "\n" for line in lines))

    return len(lines), changed
