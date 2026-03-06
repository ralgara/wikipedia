#!/usr/bin/env python3
"""Upload local pageviews JSON files to GCS with Hive partitioning.

Usage:
    python3 scripts/backfill-gcs.py /path/to/data/

Creates a temp directory with symlinks in the Hive-partitioned layout,
then uses `gsutil -m cp` for fast parallel upload. No pip dependencies needed.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BUCKET = "gs://wikipedia-cortex-data"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data-dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1]).resolve()
    files = sorted(data_dir.glob("pageviews_*.json"))
    print(f"Found {len(files)} files in {data_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        linked = 0
        for f in files:
            match = re.search(r"pageviews_(\d{4})(\d{2})(\d{2})\.json", f.name)
            if not match:
                continue
            year, month, day = match.groups()
            dest_dir = tmp / f"wikipedia/pageviews/year={year}/month={month}/day={day}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(f, dest_dir / f.name)
            linked += 1

        print(f"Prepared {linked} symlinks, uploading via gsutil -m cp...")
        result = subprocess.run(
            [
                "gsutil", "-m", "-o", "GSUtil:parallel_composite_upload_threshold=0",
                "cp", "-r", "-n",  # -n = no-clobber (skip existing)
                str(tmp / "wikipedia"),
                f"{BUCKET}/",
            ],
            text=True,
        )

        if result.returncode != 0:
            print(f"gsutil exited with code {result.returncode}")
            sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
