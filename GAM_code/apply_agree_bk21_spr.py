#!/usr/bin/env python3
"""
Replace the live `bk21_spr.csv` with a row-level agree subset produced by
`make_agree_subsets.py` (writes `bk21_data/agree_<subset>_spr.csv`).

Typical cluster prep:
  python3 make_agree_subsets.py
  python3 apply_agree_bk21_spr.py --subset all
  # then: make_bk21_jobs.py, submit LME/GAM jobs, etc.

Use --subset gpt2 (etc.) for per-model agree rows only.
"""

from __future__ import annotations

import argparse
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subset",
        default="all",
        help="Which agree file: bk21_data/agree_<subset>_spr.csv (default: all)",
    )
    ap.add_argument(
        "--agree-dir",
        default="bk21_data",
        help="Directory containing agree_*_spr.csv",
    )
    ap.add_argument(
        "--target",
        default="bk21_spr.csv",
        help="CSV that R scripts read (repo root by default)",
    )
    ap.add_argument(
        "--backup",
        default="bk21_spr_backup.csv",
        help="One-time backup of original target if backup does not exist",
    )
    args = ap.parse_args()

    src = os.path.join(args.agree_dir, "agree_%s_spr.csv" % args.subset)
    if not os.path.isfile(src):
        raise SystemExit(
            "Missing %s — run `python3 make_agree_subsets.py` first." % src
        )
    if not os.path.isfile(args.target):
        raise SystemExit("Missing target %s" % args.target)

    if not os.path.isfile(args.backup):
        shutil.copy2(args.target, args.backup)
        print("Backed up original -> %s" % args.backup)

    shutil.copy2(src, args.target)
    print("Wrote %s <- %s" % (args.target, src))


if __name__ == "__main__":
    main()
