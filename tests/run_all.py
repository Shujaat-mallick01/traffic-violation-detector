"""Run every test module in this folder. Exits non-zero if any test fails.

    python tests/run_all.py

Exists because pytest is not a project dependency — the suite is meant to run on
the same bare environment the pipeline uses.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    modules = sorted(p for p in HERE.glob("test_*.py"))
    if not modules:
        print("no test modules found")
        return 1

    failed = []
    for path in modules:
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            failed.append(path.stem)
        print()

    print("=" * 52)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(modules)} test modules passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
