#!/usr/bin/env python3
"""
Download the public cylindrical-cell battery-aging datasets referenced in
Zhang et al., "Discovery Learning accelerates battery design evaluation"
(arXiv:2508.06985), directly to an HPC scratch directory.

IMPORTANT - run this on a node with internet access.
Most HPC clusters block outbound internet on compute nodes reserved for
SLURM jobs; use a login node or a dedicated data-transfer node instead.
If your cluster requires it, run inside `screen`/`tmux` or submit as a
lightweight job on a data-transfer partition so long downloads survive
a dropped SSH session.

Confidence per source:
  - Zenodo (Samsung-25R)         : solid public REST API, should just work.
  - Figshare/KiltHub (Sony-VTC6) : figshare has a stable public API; the
                                    article ID below was parsed from the
                                    dataset DOI, double check it resolves.
  - 4TU.ResearchData (LG-MJ1)    : Cannot be downloaded with this code. Must be downloaded manually (see 2.)
  - mediaTUM (Sony-VTC5A)        : distributed via rsync with a per-dataset
                                    password shown on the landing page.
                                    Fill in NODE_ID / RSYNC_PASSWORD below.
  - Battery Archive (A123-M1A,   : Same as 4TU.ResearchData.
    LG-HG2, Sandia SNL study)      

Usage:
    python download_battery_datasets.py            # downloads everything configured
    python download_battery_datasets.py samsung_25r # downloads just one dataset
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Configuration - point this at your scratch filesystem.
# DTU HPC scratch is /work3/<username> (the newer filesystem; /work1 is
# the older one being phased out). There's no $SCRATCH env var on DTU HPC,
# so we build the path from $USER directly. Not backed up - copy final
# results to your home directory or elsewhere once done.
# --------------------------------------------------------------------------
SCRATCH = Path(f"/work3/{os.environ['USER']}")
DEST_ROOT = SCRATCH / "battery_datasets"
CHUNK = 1024 * 1024  # 1 MB streaming chunks, keeps memory flat for big files


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_download(url: str, dest_path: Path, expected_md5: str | None = None):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and expected_md5 and md5sum(dest_path) == expected_md5:
        print(f"  [skip] {dest_path.name} already present, checksum OK")
        return
    print(f"  [get ] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r        {dest_path.name}: {pct:5.1f}%", end="", flush=True)
        print()
    if expected_md5:
        got = md5sum(dest_path)
        if got != expected_md5:
            print(f"  [WARN] checksum mismatch for {dest_path.name} "
                  f"(got {got}, expected {expected_md5})")


# --------------------------------------------------------------------------
# 1. Samsung-25R -- Zhu et al., Nature Communications, hosted on Zenodo
# --------------------------------------------------------------------------
ZENODO_RECORD_ID = "6405084"  # v2 of the dataset; check zenodo.org for the latest


def fetch_samsung_25r():
    dest_dir = DEST_ROOT / "samsung_25r"
    print(f"[Samsung-25R | Zenodo record {ZENODO_RECORD_ID}]")
    meta = requests.get(
        f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}", timeout=30
    ).json()
    for f in meta["files"]:
        url = f["links"]["self"]
        name = f["key"]
        md5 = (f.get("checksum") or "").replace("md5:", "") or None
        stream_download(url, dest_dir / name, expected_md5=md5)


# --------------------------------------------------------------------------
# 2. Sony-VTC6 -- Bills et al., Scientific Data, hosted on CMU KiltHub
#    (KiltHub runs on Figshare, which has a stable public API)
# --------------------------------------------------------------------------

# Must be dowloaded manually at 
# https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830?file=26855063
# and tranfered via sftp to the HPC cluster (use username@transfer.gbar.dtu.dk)

# --------------------------------------------------------------------------
# 3. LG-MJ1 -- Trad, VITO/EnergyVille, hosted on 4TU.ResearchData
# --------------------------------------------------------------------------
FOURTU_DOI = "10.4121/13739296"
 
# Paste the "download entire dataset" link from the 4TU dataset page here.
FOURTU_BULK_URL = "https://data.4tu.nl/ndownloader/items/e19fe272-4f46-450c-9125-6545c4c1a98b/versions/1"
FOURTU_BULK_FILENAME = "lg_mj1_full_dataset.zip"
 
# Fallback: if 4TU only gives you per-file links instead of one bulk zip,
# list them here and fetch_lg_mj1() will use these instead.
DIRECT_4TU_FILES: list[tuple[str, str]] = [
    # ("https://data.4tu.nl/.../lg_mj1_cell01.csv", "lg_mj1_cell01.csv"),
]
 
 
def fetch_lg_mj1():
    dest_dir = DEST_ROOT / "lg_mj1"
    print(f"[LG-MJ1 | 4TU.ResearchData {FOURTU_DOI}]")
    if FOURTU_BULK_URL:
        stream_download(FOURTU_BULK_URL, dest_dir / FOURTU_BULK_FILENAME)
        return
    if DIRECT_4TU_FILES:
        for url, name in DIRECT_4TU_FILES:
            stream_download(url, dest_dir / name)
        return
    landing = requests.get(
        f"https://doi.org/{FOURTU_DOI}", timeout=30, allow_redirects=True
    )
    print(f"  [WARN] No download URL configured yet.")
    print(f"         DOI resolved to: {landing.url}")
    print("         Open that page, copy either the 'download entire dataset' "
          "link into FOURTU_BULK_URL, or individual file links into "
          "DIRECT_4TU_FILES, in this script.")
 


# --------------------------------------------------------------------------
# 4. Sony-VTC5A -- Wildfeuer et al., hosted on TUM mediaTUM (rsync only)
# --------------------------------------------------------------------------
MEDIATUM_NODE_ID = "1713382"  # e.g. "1750435" -- get this from the dataset's page
MEDIATUM_RSYNC_PASSWORD = "m1713382"  # shown on the same page


def fetch_sony_vtc5a():
    dest_dir = DEST_ROOT / "sony_vtc5a"
    print("[Sony-VTC5A | TUM mediaTUM]")
    if not MEDIATUM_NODE_ID:
        print("  [WARN] MEDIATUM_NODE_ID not set. Find the Wildfeuer et al. "
              "dataset on mediatum.ub.tum.de, open its page, and copy the "
              "module ID + rsync password shown under 'Technische Hinweise'.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-avzP",
        f"rsync://m{MEDIATUM_NODE_ID}@dataserv.ub.tum.de/m{MEDIATUM_NODE_ID}/",
        str(dest_dir) + "/",
    ]
    print("  " + " ".join(cmd))
    if MEDIATUM_RSYNC_PASSWORD:
        env = os.environ.copy()
        env["RSYNC_PASSWORD"] = MEDIATUM_RSYNC_PASSWORD
        subprocess.run(cmd, env=env, check=True)
    else:
        print("  (no password set - rsync will prompt interactively)")
        subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# 5. A123-M1A / LG-HG2 -- Preger et al., Sandia National Labs,
#    hosted on batteryarchive.org (no documented bulk API)
# --------------------------------------------------------------------------

    # Must be downloaded manually from
    # https://data.mendeley.com/datasets/sn2kv34r4h/2
    # and then transfered same as 2.


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
DATASETS = {
    "samsung_25r": fetch_samsung_25r,
    "lg_mj1": fetch_lg_mj1,
    "sony_vtc5a": fetch_sony_vtc5a,
}

if __name__ == "__main__":
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into: {DEST_ROOT}\n")
    
    
    requested = sys.argv[1:] or list(DATASETS.keys())
    for name in requested:
        if name not in DATASETS:
            print(f"Unknown dataset '{name}'. Choices: {list(DATASETS.keys())}")
            continue
        DATASETS[name]()
        print()

    print("Done. Contents of", DEST_ROOT, ":")
    
    for p in sorted(DEST_ROOT.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(DEST_ROOT), f"({p.stat().st_size / 1e6:.1f} MB)")
    