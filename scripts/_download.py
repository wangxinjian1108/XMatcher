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


def _http_headers(url: str) -> dict[str, str]:
    """Headers to attach to every HTTP fetch.

    User-Agent is set unconditionally (default Python-urllib UA, and curl's
    default UA when called by Docker layers, are sometimes 404'd by CDN
    edge caches). HF_TOKEN is attached as Bearer for any huggingface.co
    host (including subdomains like cdn-lfs.huggingface.co).
    """
    headers: dict[str, str] = {
        "User-Agent": "xmatcher-download/0.1 (+https://github.com/wangxinjian1108/XMatcher)",
    }
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("huggingface.co"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_http(url: str, dest: Path) -> None:
    """Download `url` to `dest` via curl.

    Why curl rather than urllib: HuggingFace's /resolve/ URLs return 302
    redirects to per-request signed CDN URLs (cdn-lfs-*.huggingface.co).
    Python's urllib follows redirects but strips Authorization across
    origins, and some path/UA combinations return 404 instead of the
    expected 302 — a known wart of urlopen against modern S3-fronted CDNs.
    curl's redirect + retry handling is battle-tested and works the same
    way HuggingFace's own client libraries do.
    """
    import shutil
    import subprocess
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl") is None:
        raise RuntimeError(
            "curl not found in PATH; needed by xmatcher's weight downloader."
        )
    header_args: list[str] = []
    for k, v in _http_headers(url).items():
        header_args += ["-H", f"{k}: {v}"]
    cmd = [
        "curl", "-fL", "--retry", "3", "--retry-delay", "2",
        *header_args, "-o", str(dest), url,
    ]
    subprocess.run(cmd, check=True)


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
