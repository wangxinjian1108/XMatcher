#!/usr/bin/env python
"""Download weights described in weights/WEIGHTS.lock.

Reads the lock file, fetches missing/changed entries for the given method,
and verifies sha256. Hard error on mismatch — never silent success.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
import yaml


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_gdrive(file_id: str, dest: Path) -> None:
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)


def _fetch_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _process_entry(name: str, entry: dict, target_root: Path) -> None:
    target = target_root / entry["target"]
    expected_sha = entry.get("sha256", "")
    if expected_sha.startswith("<"):
        sys.exit(
            f"[download] '{name}': sha256 placeholder not filled in WEIGHTS.lock. "
            f"Run a one-time manual download, then update the lock file."
        )
    if target.exists() and _sha256_of(target) == expected_sha:
        print(f"[download] {name}: up to date ({target})")
        return
    if "gdrive_id" in entry:
        gid = entry["gdrive_id"]
        if gid.startswith("<"):
            sys.exit(f"[download] '{name}': gdrive_id placeholder not filled.")
        _fetch_gdrive(gid, target)
    elif "http_url" in entry:
        _fetch_http(entry["http_url"], target)
    else:
        sys.exit(f"[download] '{name}': missing 'gdrive_id' or 'http_url'.")
    actual = _sha256_of(target)
    if actual != expected_sha:
        target.unlink(missing_ok=True)
        sys.exit(
            f"[download] '{name}' sha256 mismatch: expected {expected_sha}, got {actual}"
        )
    print(f"[download] {name}: ok → {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()

    lock_path = Path(__file__).resolve().parents[1] / "weights" / "WEIGHTS.lock"
    lock = yaml.safe_load(lock_path.read_text()) or {}
    if args.method not in lock:
        sys.exit(f"[download] no entries for method '{args.method}' in {lock_path}")
    for variant, entry in lock[args.method].items():
        _process_entry(f"{args.method}.{variant}", entry, args.target)


if __name__ == "__main__":
    main()
