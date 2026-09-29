"""Delete local vector/doc stores so the next ingestion starts from scratch.

Usage:
    python scripts/clean_store.py          # preview (dry run)
    python scripts/clean_store.py --force  # actually delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402


def remove(path: Path, force: bool) -> None:
    if not path.exists():
        print(f"Skipping: {path} (does not exist)")
        return
    if force:
        shutil.rmtree(path)
        print(f"Deleted: {path}")
    else:
        print(f"Would delete: {path}  (re-run with --force)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean local stores")
    parser.add_argument("--force", action="store_true", help="Actually delete files.")
    args = parser.parse_args()

    settings = get_settings()
    targets = [
        settings.local_store_path,
        settings.vector_store_path,
    ]
    for target in targets:
        remove(target, args.force)
    if not args.force:
        print("\nDry run only. Re-run with --force to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
