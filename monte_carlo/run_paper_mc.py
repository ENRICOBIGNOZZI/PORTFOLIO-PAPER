#!/usr/bin/env python3
"""Reproduce the paper Monte Carlo exactly."""
import sys
from finance_mc_run import main

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--reps", "160",
        "--rate-reps", "100",
        "--seed", "20260818",
        "--ridge-scale", "2.0",
        "--jobs", "-1",
    ]
    main()
