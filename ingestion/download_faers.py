"""
ingestion/download_faers.py
============================
Download every FDA FAERS (Adverse Event Reporting System) quarterly file
from openFDA.

Source manifest:
    https://api.fda.gov/download.json

Each entry in the manifest looks like:
    {
      "file": "https://download.open.fda.gov/drug/event/2019q1/drug-event-0001-of-0007.json.zip",
      "size_mb": 197.4,
      "records": 248712,
      ...
    }

Usage:
    python ingestion/download_faers.py                       # download all
    python ingestion/download_faers.py --from-year 2018      # only newer
    python ingestion/download_faers.py --quarter 2024q1      # one quarter
    python ingestion/download_faers.py --limit 2             # quick test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

LOG = logging.getLogger("download_faers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

OPENFDA_MANIFEST = "https://api.fda.gov/download.json"
DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def fetch_manifest(session: requests.Session) -> dict:
    """Return the openFDA download manifest JSON."""
    LOG.info("Fetching openFDA download manifest …")
    resp = session.get(OPENFDA_MANIFEST, timeout=60)
    resp.raise_for_status()
    return resp.json()


def iter_drug_event_partitions(manifest: dict) -> Iterable[dict]:
    """
    The manifest has nested structure:
        results -> drug -> event -> partitions: [ {file, size_mb, records, display_name} ]
    Yield every partition for drug/event.
    """
    partitions = (
        manifest.get("results", {}).get("drug", {}).get("event", {}).get("partitions", [])
    )
    if not partitions:
        raise RuntimeError("No drug/event partitions found in manifest")
    for p in partitions:
        yield p


def _quarter_from_url(url: str) -> str:
    # ".../drug/event/2019q1/drug-event-0001-of-0007.json.zip" -> "2019q1"
    parts = url.rstrip("/").split("/")
    for token in reversed(parts):
        if "q" in token and token[:4].isdigit():
            return token.lower()
    return "unknown"


def _year_from_quarter(quarter: str) -> int:
    try:
        return int(quarter[:4])
    except Exception:
        return 0


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(session: requests.Session, url: str, dest: Path, retries: int = 3) -> Path:
    """Stream-download with progress bar; resume-friendly via .part suffix."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        LOG.info("[skip] %s already present (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with tmp.open("wb") as fp, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        fp.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Attempt %d failed for %s: %s", attempt, url, exc)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} retries")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download FDA FAERS drug/event files")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR,
                        help="Destination directory (default: data/raw)")
    parser.add_argument("--from-year", type=int, default=0,
                        help="Only download files from this year onward")
    parser.add_argument("--quarter", type=str, default=None,
                        help="Only download a single quarter, e.g. 2024q1")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N files (handy for smoke tests)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest = fetch_manifest(session)

    candidates = []
    for p in iter_drug_event_partitions(manifest):
        url = p["file"]
        quarter = _quarter_from_url(url)
        year = _year_from_quarter(quarter)
        if args.from_year and year < args.from_year:
            continue
        if args.quarter and quarter != args.quarter.lower():
            continue
        candidates.append((quarter, url, p))

    if not candidates:
        LOG.error("No matching partitions; check --from-year / --quarter")
        return 1

    def _size_mb(p: dict) -> float:
        """openFDA size_mb is a string like '197.4' — coerce safely."""
        try:
            return float(p.get("size_mb") or 0)
        except (TypeError, ValueError):
            return 0.0

    LOG.info("Will download %d files (%.1f GB total)",
             len(candidates),
             sum(_size_mb(p) for _, _, p in candidates) / 1024)
    manifest_log = args.out_dir / "_manifest.jsonl"
    with manifest_log.open("a") as log_fp:
        for i, (quarter, url, meta) in enumerate(candidates, start=1):
            if args.limit and i > args.limit:
                break
            qdir = args.out_dir / quarter
            dest = qdir / Path(url).name
            LOG.info("[%d/%d] %s -> %s", i, len(candidates), url, dest)
            download_one(session, url, dest)
            log_fp.write(json.dumps({
                "quarter": quarter,
                "url": url,
                "local_path": str(dest),
                "size_bytes": dest.stat().st_size,
                "records_hint": meta.get("records"),
            }) + "\n")

    LOG.info("All done. Files under %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
