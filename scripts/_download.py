#!/usr/bin/env python
"""Download weights described in weights/WEIGHTS.lock.

Reads the lock file, fetches missing/changed entries for the given method,
and verifies sha256. Hard error on mismatch — never silent success.

Bootstrap mode (--bootstrap):
    For entries whose sha256 is "BOOTSTRAP" (or starts with "<"), download
    the file regardless and print its computed sha256 so you can paste it
    back into WEIGHTS.lock. Without --bootstrap, those entries hard-error.

HuggingFace gated/private repos:
    When HF_TOKEN env var is set, an Authorization: Bearer header is sent
    on requests to huggingface.co. Public files don't need this.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
import yaml


_BOOTSTRAP_SENTINELS = ("BOOTSTRAP",)


def _is_placeholder(value: str) -> bool:
    return value in _BOOTSTRAP_SENTINELS or value.startswith("<")


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


def _build_http_request(url: str) -> urllib.request.Request:
    """Build a Request with a real User-Agent. HF (and some CDNs) return 404 to
    the default Python-urllib/X UA. HF_TOKEN is attached as Bearer for any
    huggingface.co host (including subdomains like cdn-lfs.huggingface.co).
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "xmatcher-download/0.1 (+https://github.com/wangxinjian1108/XMatcher)"},
    )
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("huggingface.co"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
    return req


def _fetch_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = _build_http_request(url)
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _do_download(name: str, entry: dict, target: Path) -> None:
    if "gdrive_id" in entry:
        gid = entry["gdrive_id"]
        if _is_placeholder(gid):
            sys.exit(f"[download] '{name}': gdrive_id placeholder not filled.")
        _fetch_gdrive(gid, target)
    elif "http_url" in entry:
        _fetch_http(entry["http_url"], target)
    else:
        sys.exit(f"[download] '{name}': missing 'gdrive_id' or 'http_url'.")


def _process_entry(name: str, entry: dict, target_root: Path, *, bootstrap: bool) -> None:
    target = target_root / entry["target"]
    expected_sha = entry.get("sha256", "")

    if _is_placeholder(expected_sha):
        if not bootstrap:
            sys.exit(
                f"[download] '{name}': sha256 placeholder ({expected_sha!r}) in WEIGHTS.lock. "
                f"Run with --bootstrap to download and capture the real sha256."
            )
        _do_download(name, entry, target)
        actual = _sha256_of(target)
        print()
        print(f"[bootstrap] {name}")
        print(f"  downloaded → {target}")
        print(f"  sha256:      {actual}")
        print(f"  ACTION: paste this sha256 into weights/WEIGHTS.lock under '{name}'.")
        return

    if target.exists() and _sha256_of(target) == expected_sha:
        print(f"[download] {name}: up to date ({target})")
        return
    _do_download(name, entry, target)
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
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Allow downloading entries whose sha256 is a placeholder; "
             "print the captured sha256 so you can update WEIGHTS.lock.",
    )
    args = parser.parse_args()

    lock_path = Path(__file__).resolve().parents[1] / "weights" / "WEIGHTS.lock"
    lock = yaml.safe_load(lock_path.read_text()) or {}
    if args.method not in lock:
        sys.exit(f"[download] no entries for method '{args.method}' in {lock_path}")
    for variant, entry in lock[args.method].items():
        _process_entry(
            f"{args.method}.{variant}", entry, args.target, bootstrap=args.bootstrap
        )


if __name__ == "__main__":
    main()
