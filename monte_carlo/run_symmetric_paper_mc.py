#!/usr/bin/env python3
"""Run the final symmetric Monte Carlo layer.

The validated baseline Sobolev-SNR, Sobolev-N, and d=3 CSVs are distributed in
the release ZIP. This entry point generates the new linear SNR/N and d=1,2,3
cells, then rebuilds the matrix figures, separate history-required figures, and
the main summary table.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    py = sys.executable

    if not args.skip_simulation:
        command = [py, "symmetric_comparative_mc.py", "--jobs", str(args.jobs),
                   "--seed", str(args.seed)]
        if args.quick:
            command.append("--quick")
        if args.resume:
            command.append("--resume")
        run(command)
    if args.quick:
        return
    run([py, "postprocess_symmetric_mc.py", "--bootstrap", "2000",
         "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
