#!/usr/bin/env python3
"""Run the repository Monte Carlo and the final comparative-statistics extension."""
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
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--skip-base", action="store_true",
                        help="Do not rerun the existing linear/nonlinear paper simulation.")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Reuse included comparative-statistics CSVs and rebuild summaries only.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_base and not args.skip_simulation:
        run([py, "run_paper_mc.py"])

    if not args.skip_simulation:
        command = [py, "comparative_statics_mc.py", "--jobs", str(args.jobs),
                   "--seed", str(args.seed)]
        if args.quick:
            command.append("--quick")
        run(command)

    run([py, "postprocess_comparative_statics.py", "--bootstrap", "2000",
         "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
