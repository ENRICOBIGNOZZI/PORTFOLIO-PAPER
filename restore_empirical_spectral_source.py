#!/usr/bin/env python3
"""Restore the frozen empirical and spectral source trees from the repository bundle."""
from pathlib import Path
import hashlib
import zipfile

BUNDLE = Path("bundles/PORTFOLIO_PAPER_EMPIRICAL_SPECTRAL_CODE.zip")
EXPECTED_SHA256 = "1a8982011fe2fc8bbe554e2fbdae90717a3bd90124356c479769d1b1f1edec99"

payload = BUNDLE.read_bytes()
digest = hashlib.sha256(payload).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f"bundle SHA256 mismatch: {digest}")
with zipfile.ZipFile(BUNDLE) as archive:
    archive.extractall(".")
print(f"Restored empirical/ and spectral_decay/; SHA256={digest}")
