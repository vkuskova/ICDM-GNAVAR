#!/usr/bin/env python3
"""
verify_paper_numbers.py
=======================
Recompute every numerical claim in the paper from the committed results artifacts
and check it against the value stated in the paper.

Usage
-----
    python verify_paper_numbers.py
    python verify_paper_numbers.py --root /path/to/repo   # if run from elsewhere

Exit code 0 if all checks pass, 1 otherwise. No GPU required; runs in ~seconds.
This is "Path A" (verify committed artifacts). To regenerate the artifacts from
scratch, see notebooks/reproduce.ipynb ("Path B"), which requires a GPU.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from verifier_core import CHECKS  # noqa: E402


def _compare(kind, expected, computed, tol):
    if kind == "exact":
        return computed == expected
    if kind == "tol":
        return abs(float(computed) - float(expected)) <= tol
    if kind == "range":
        lo, hi = expected
        # computed may be a scalar or a (min, max) tuple that must lie within [lo, hi]
        if isinstance(computed, (tuple, list)):
            return lo <= computed[0] and computed[1] <= hi
        return lo <= computed <= hi
    raise ValueError(f"unknown kind {kind}")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    root = args.root

    n_pass = 0
    n_fail = 0
    failures = []

    print("=" * 78)
    print("Reproducibility verification: recomputing every paper number from artifacts")
    print("=" * 78)
    cur_section = None
    for entry in CHECKS:
        section, quantity, kind, expected, fn = entry[:5]
        tol = entry[5] if len(entry) > 5 else 0.0
        if section != cur_section:
            print(f"\n[Section {section}]")
            cur_section = section
        try:
            computed = fn(root)
            ok = _compare(kind, expected, computed, tol)
        except Exception as e:  # noqa: BLE001
            computed = f"ERROR: {e}"
            ok = False
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((section, quantity, expected, computed))
        extra = f" (tol {tol})" if kind == "tol" else ""
        print(f"  [{status}] {quantity:<52} paper={_fmt(expected)}{extra}  artifact={_fmt(computed)}")

    print("\n" + "=" * 78)
    print(f"RESULT: {n_pass}/{n_pass + n_fail} checks passed.")
    if n_fail:
        print(f"\n{n_fail} FAILED:")
        for s, q, e, c in failures:
            print(f"  [{s}] {q}: paper={_fmt(e)} artifact={_fmt(c)}")
        print("=" * 78)
        sys.exit(1)
    print("All paper numbers reproduce from the committed artifacts.")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
