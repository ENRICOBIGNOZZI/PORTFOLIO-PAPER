#!/usr/bin/env python3
"""Entry point for symmetric Monte Carlo post-processing."""
from pathlib import Path

_PARTS = Path(__file__).resolve().parent / "postprocess_parts"
for _part in sorted(_PARTS.glob("part*.py")):
    exec(compile(_part.read_text(), str(_part), "exec"), globals(), globals())
