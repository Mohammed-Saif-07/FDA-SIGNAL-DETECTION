"""
ingestion/load_to_hdfs.py
=========================
Upload the local Parquet partitions produced by parse_faers.py into HDFS
so Hive can read them via external tables.

Layout in HDFS:
    /user/hive/warehouse/fda_pharma.db/raw_adverse_events/
        report_year=YYYY/report_quarter=Q/part-*.parquet

Two transport modes:
    1) WebHDFS via the `hdfs` Python client    (default — pure Python, no java)
    2) Native `hdfs dfs -put`                  (used inside docker namenode)

Usage:
    python ingestion/load_to_hdfs.py
    python ingestion/load_to_hdfs.py --mode shell      # use hdfs dfs
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("load_to_hdfs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PARQUET = ROOT / "data" / "processed" / "raw_adverse_events"
HDFS_TARGET = "/user/hive/warehouse/fda_pharma.db/raw_adverse_events"
WEBHDFS_URL = os.getenv("WEBHDFS_URL", "http://localhost:9870")
HDFS_USER = os.getenv("HDFS_USER", "root")


def upload_webhdfs(local: Path, remote_root: str) -> None:
    from hdfs import InsecureClient  # type: ignore

    LOG.info("Connecting to WebHDFS at %s as %s", WEBHDFS_URL, HDFS_USER)
    client = InsecureClient(WEBHDFS_URL, user=HDFS_USER)
    client.makedirs(remote_root)

    for fpath in local.rglob("*"):
        if fpath.is_dir():
            continue
        rel = fpath.relative_to(local).as_posix()
        remote = f"{remote_root}/{rel}"
        LOG.info("PUT %s -> %s", fpath, remote)
        client.makedirs(os.path.dirname(remote))
        client.upload(remote, str(fpath), overwrite=True)


def upload_shell(local: Path, remote_root: str) -> None:
    if not shutil.which("hdfs"):
        raise RuntimeError("hdfs CLI not on PATH — run inside the namenode container")
    subprocess.check_call(["hdfs", "dfs", "-mkdir", "-p", remote_root])
    subprocess.check_call(["hdfs", "dfs", "-put", "-f", str(local) + "/.", remote_root])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, default=LOCAL_PARQUET)
    p.add_argument("--remote", type=str, default=HDFS_TARGET)
    p.add_argument("--mode", choices=["webhdfs", "shell"], default="webhdfs")
    args = p.parse_args()

    if not args.local.exists():
        LOG.error("Local parquet directory missing: %s — run parse_faers.py first",
                  args.local)
        return 1

    if args.mode == "webhdfs":
        upload_webhdfs(args.local, args.remote)
    else:
        upload_shell(args.local, args.remote)

    LOG.info("Upload complete -> hdfs://%s", args.remote)
    return 0


if __name__ == "__main__":
    sys.exit(main())
