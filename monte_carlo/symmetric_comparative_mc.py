#!/usr/bin/env python3
"""Entry point for the symmetric Linear/Sobolev Monte Carlo.

The implementation is split into readable top-level source parts to keep GitHub
connector commits small. The exact monolithic source is included in the release ZIP.
"""
from pathlib import Path

_PARTS = Path(__file__).resolve().parent / "symmetric_parts"
for _part in sorted(_PARTS.glob("part*.py")):
    exec(compile(_part.read_text(), str(_part), "exec"), globals(), globals())
