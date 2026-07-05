#!/usr/bin/env python3
"""
Download twanksta_entries.json from the latest strfry/prussian-corpus release.

Verifies SHA256 hash against the release asset metadata.
Idempotent: skips if the file already exists with the correct hash.

Usage:
  python3 scripts/download_twanksta_entries.py
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "strfry/prussian-corpus"
ASSET_NAME = "twanksta_entries.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / ASSET_NAME

RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases/latest"


def get_release_asset() -> dict | None:
    """Fetch the latest release metadata and find our asset."""
    req = urllib.request.Request(RELEASES_URL, headers={
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "prussian-mcp-downloader/1.0",
    })
    with urllib.request.urlopen(req) as resp:
        release = json.loads(resp.read())

    for asset in release.get("assets", []):
        if asset["name"] == ASSET_NAME:
            return asset
    return None


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_asset(url: str, target: Path):
    """Download asset, streaming to disk."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "prussian-mcp-downloader/1.0",
    })
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(target, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading: {downloaded//1024**2}MB/{total//1024**2}MB ({pct:.0f}%)", end="")
    print()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching latest release from {REPO}...")
    asset = get_release_asset()
    if not asset:
        print(f"Error: Asset '{ASSET_NAME}' not found in latest release.", file=sys.stderr)
        sys.exit(1)

    expected_hash = asset.get("digest", "").removeprefix("sha256:")
    url = asset["browser_download_url"]
    size_mb = asset["size"] / 1024**2

    print(f"  Asset:  {ASSET_NAME}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  SHA256: {expected_hash}")
    print(f"  URL:    {url}")

    # Check if already downloaded with correct hash
    if OUTPUT_FILE.exists():
        existing_hash = sha256_file(OUTPUT_FILE)
        if existing_hash == expected_hash:
            print(f"\n✓ {OUTPUT_FILE} already up-to-date.")
            return
        print(f"\n  Hash mismatch (got {existing_hash}), re-downloading...")

    print(f"\nDownloading to {OUTPUT_FILE}...")
    download_asset(url, OUTPUT_FILE)

    actual_hash = sha256_file(OUTPUT_FILE)
    if actual_hash != expected_hash:
        print(f"Error: SHA256 mismatch after download!", file=sys.stderr)
        print(f"  Expected: {expected_hash}", file=sys.stderr)
        print(f"  Got:      {actual_hash}", file=sys.stderr)
        OUTPUT_FILE.unlink(missing_ok=True)
        sys.exit(1)

    print(f"✓ Download complete — {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
