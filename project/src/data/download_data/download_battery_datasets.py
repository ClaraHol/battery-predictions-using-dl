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

import csv
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ===========================================================================
# Configuration - point this at your scratch filesystem.
# DTU HPC scratch is /work3/<username> Not backed up - copy final
# results to your home directory or elsewhere once done.
# ===========================================================================
SCRATCH = Path(f"/work3/{os.environ['USER']}")
DEST_ROOT = SCRATCH / "battery_datasets"
CHUNK = 1024 * 1024  # 1 MB streaming chunks, keeps memory flat for big files


# ===========================================================================
# Shared helpers
# ===========================================================================
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
                    print(
                        f"\r        {dest_path.name}: {pct:5.1f}%", end="", flush=True
                    )
        print()
    if expected_md5:
        got = md5sum(dest_path)
        if got != expected_md5:
            print(
                f"  [WARN] checksum mismatch for {dest_path.name} "
                f"(got {got}, expected {expected_md5})"
            )


# ===========================================================================
# 1. Samsung-25R -- Zhu et al., Nature Communications, hosted on Zenodo
# ===========================================================================
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


# ===========================================================================
# 2. Sony-VTC6 -- Bills et al., Scientific Data, hosted on CMU KiltHub
#    (KiltHub runs on Figshare, which has a stable public API)
# ===========================================================================

# Must be dowloaded manually at
# https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830?file=26855063
# and tranfered via sftp to the HPC cluster (use username@transfer.gbar.dtu.dk)

# ===========================================================================
# 3. LG-MJ1 -- Trad, VITO/EnergyVille, hosted on 4TU.ResearchData
# ===========================================================================
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
    print("  [WARN] No download URL configured yet.")
    print(f"         DOI resolved to: {landing.url}")
    print(
        "         Open that page, copy either the 'download entire dataset' "
        "link into FOURTU_BULK_URL, or individual file links into "
        "DIRECT_4TU_FILES, in this script."
    )


# ===========================================================================
# 4. Sony-VTC5A -- Wildfeuer et al., hosted on TUM mediaTUM (rsync only)
# ===========================================================================
MEDIATUM_NODE_ID = "1713382"  # e.g. "1750435" -- get this from the dataset's page
MEDIATUM_RSYNC_PASSWORD = "m1713382"  # shown on the same page


def fetch_sony_vtc5a():
    dest_dir = DEST_ROOT / "sony_vtc5a"
    print("[Sony-VTC5A | TUM mediaTUM]")
    if not MEDIATUM_NODE_ID:
        print(
            "  [WARN] MEDIATUM_NODE_ID not set. Find the Wildfeuer et al. "
            "dataset on mediatum.ub.tum.de, open its page, and copy the "
            "module ID + rsync password shown under 'Technische Hinweise'."
        )
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-avzP",
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


# ===========================================================================
# 5. A123-M1A / LG-HG2 -- Preger et al., Sandia National Labs,
#    hosted on batteryarchive.org (no documented bulk API)
# ===========================================================================

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_URL = "https://database.batteryarchive.org/api/queries"

# Your public Battery Archive dashboard key
PUBLIC_KEY = "Key 61PzyYn9u56njtPDOh37VVBFasGusUlxc7WZ1FSj"


PROGRESS_FILE = Path("battery_data/progress.csv")

MAX_CYCLE = 100
BLOCK_SIZE = 3

MAX_RETRIES = 5
POLL_INTERVAL = 2
MAX_POLL_ATTEMPTS = 60

# All 25 cells
CELL_IDS = [
    # LFP
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_a",
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_b",
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_c",
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_d",
    "SNL_18650_LFP_25C_20-80_0.5/3C_a",
    "SNL_18650_LFP_25C_40-60_0.5/0.5C_a",
    "SNL_18650_LFP_25C_40-60_0.5/0.5C_b",
    "SNL_18650_LFP_25C_40-60_0.5/3C_a",
    "SNL_18650_LFP_25C_40-60_0.5/3C_b",
    # # NCA
    # "SNL_18650_NCA_25C_20-80_0.5/0.5C_a",
    # "SNL_18650_NCA_25C_20-80_0.5/0.5C_b",
    # "SNL_18650_NCA_25C_20-80_0.5/0.5C_c",
    # "SNL_18650_NCA_25C_20-80_0.5/0.5C_d",
    # "SNL_18650_NCA_25C_40-60_0.5/0.5C_a",
    # "SNL_18650_NCA_25C_40-60_0.5/0.5C_b",
    # # NMC
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_a",
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_b",
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_c",
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_d",
    "SNL_18650_NMC_25C_20-80_0.5/3C_a",
    "SNL_18650_NMC_25C_20-80_0.5/3C_b",
    "SNL_18650_NMC_25C_40-60_0.5/0.5C_a",
    "SNL_18650_NMC_25C_40-60_0.5/0.5C_b",
    "SNL_18650_NMC_25C_40-60_0.5/3C_a",
    "SNL_18650_NMC_25C_40-60_0.5/3C_b",
]


HEADERS = {
    "Authorization": PUBLIC_KEY,
    "Content-Type": "application/json",
}

# --------------------------------------------------
# Query function
# --------------------------------------------------


def run_query(query_id, parameters):
    payload = {
        "id": query_id,
        "parameters": parameters,
        "apply_auto_limit": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{BASE_URL}/{query_id}/results",
                headers=HEADERS,
                json=payload,
                timeout=300,
            )

            response.raise_for_status()
            result = response.json()

            # ----------------------------------------
            # Immediate result
            # ----------------------------------------
            if "query_result" in result:
                return result["query_result"]["data"]

            # ----------------------------------------
            # Asynchronous job
            # ----------------------------------------
            if "job" not in result:
                raise RuntimeError(f"Unexpected response:\n{result}")

            job_id = result["job"]["id"]
            print(f"        Async job: {job_id}")

            for poll_attempt in range(MAX_POLL_ATTEMPTS):
                time.sleep(POLL_INTERVAL)

                job_response = requests.get(
                    f"https://database.batteryarchive.org/api/jobs/{job_id}",
                    headers=HEADERS,
                    timeout=60,
                )

                job_response.raise_for_status()

                job = job_response.json()["job"]
                status = job["status"]

                if status == 3:
                    query_result_id = job["query_result_id"]
                    print(f"        Job finished (result {query_result_id})")
                    result_response = requests.get(
                        f"https://database.batteryarchive.org/api/query_results/{query_result_id}",
                        headers=HEADERS,
                        timeout=300,
                    )
                    result_response.raise_for_status()
                    result = result_response.json()

                    return result["query_result"]["data"]

                elif status == 4:
                    raise RuntimeError(f"Query failed: {job.get('error')}")

                elif status == 5:
                    raise RuntimeError("Query was cancelled")

            raise TimeoutError(f"Job {job_id} did not finish")

        except Exception as e:
            print(f"        Attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt < MAX_RETRIES:
                wait_time = 5 * attempt
                print(f"        Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise


# --------------------------------------------------
# Progress handling
# --------------------------------------------------


def load_progress():

    if not PROGRESS_FILE.exists():
        return {}

    progress = {}
    with open(
        PROGRESS_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            progress[row["cell_id"]] = {
                "last_cycle": int(row["last_cycle"]),
                "status": row["status"],
            }
    return progress


def save_progress(progress):

    PROGRESS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = PROGRESS_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id",
                "last_cycle",
                "status",
            ],
        )

        writer.writeheader()

        for cell_id, info in progress.items():
            writer.writerow(
                {
                    "cell_id": cell_id,
                    "last_cycle": info["last_cycle"],
                    "status": info["status"],
                }
            )

    # Atomic replacement
    temp_file.replace(PROGRESS_FILE)


# ============================================================
# Main downloader for Sandia SNL
# ============================================================


def fetch_sandia_snl():

    dest_dir = DEST_ROOT / "sandia_snl"
    progress = load_progress()

    for i, cell_id in enumerate(CELL_IDS, start=1):
        print()
        print("=" * 80)
        print(f"[{i}/{len(CELL_IDS)}] {cell_id}")
        print("=" * 80)

        # ----------------------------------------------------
        # Check whether this cell was already completed
        # ----------------------------------------------------

        if cell_id in progress and progress[cell_id]["status"] == "complete":
            print("    Already complete. Skipping.")
            continue

        # ----------------------------------------------------
        # Determine where to resume
        # ----------------------------------------------------

        if cell_id in progress:
            last_cycle = progress[cell_id]["last_cycle"]
            start_cycle = last_cycle + 1
            print(f"    Resuming after cycle {last_cycle}")

        else:
            start_cycle = 1
            print("    Starting from cycle 1")

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        # Replace "/" because it would otherwise create
        # subdirectories.
        safe_name = cell_id.replace("/", "_")
        output_file = dest_dir / f"{safe_name}.csv"

        # ----------------------------------------------------
        # Determine whether we append or create
        # ----------------------------------------------------

        file_exists = output_file.exists()

        with open(
            output_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "cell_id",
                    "cycle",
                    "cycle_time",
                    "v",
                    "row_m",
                    "label",
                    "row",
                ],
            )

            if not file_exists:
                writer.writeheader()

            # ------------------------------------------------
            # Cycle loop
            # ------------------------------------------------

            for j in range(start_cycle, MAX_CYCLE + 1, BLOCK_SIZE):
                cycle_1 = j
                cycle_2 = j + 1
                cycle_3 = j + 2

                print(f"    Cycles {cycle_1}-{cycle_3}")

                # --------------------------------------------
                # Query
                # --------------------------------------------

                try:
                    data = run_query(
                        query_id=30,
                        parameters={
                            "cell_id": [cell_id],
                            "cycle_1": cycle_1,
                            "cycle_2": cycle_2,
                            "cycle_3": cycle_3,
                        },
                    )

                except Exception as e:
                    print()
                    print(f"    Could not retrieve cycles {cycle_1}-{cycle_3}")
                    print(f"    Error: {e}")
                    print("    Stopping this cell.")

                    progress[cell_id] = {
                        "last_cycle": j - 1,
                        "status": "failed",
                    }

                    save_progress(progress)
                    break

                rows = data["rows"]

                # --------------------------------------------
                # No data
                # --------------------------------------------

                if not rows:
                    print("        No rows returned.")
                    print(f"        End of data around cycle {cycle_1}.")

                    progress[cell_id] = {
                        "last_cycle": j - 1,
                        "status": "complete",
                    }

                    save_progress(progress)
                    break

                # --------------------------------------------
                # Write data
                # --------------------------------------------

                for row in rows:
                    label = row.get(
                        "label",
                        "",
                    )

                    # Extract cycle from label
                    cycle = label.rsplit(" ", 1)[-1] if label else None

                    writer.writerow(
                        {
                            "cell_id": cell_id,
                            "cycle": cycle,
                            "cycle_time": row.get("cycle_time"),
                            "v": row.get("v"),
                            "row_m": row.get("row_m"),
                            "label": label,
                            "row": row.get("row"),
                        }
                    )

                # --------------------------------------------
                # Force data to disk
                # --------------------------------------------

                csvfile.flush()

                # --------------------------------------------
                # Checkpoint
                # --------------------------------------------

                last_cycle_this_block = min(j + BLOCK_SIZE - 1, MAX_CYCLE)

                progress[cell_id] = {
                    "last_cycle": last_cycle_this_block,
                    "status": "in_progress",
                }

                save_progress(progress)

                print(f"        {len(rows):,} rows")
                print(f"        Checkpoint: cycle {last_cycle_this_block}")

                # Don't hammer Battery Archive
                time.sleep(0.5)

            else:
                # The for loop reached MAX_CYCLE
                # without breaking.

                progress[cell_id] = {"last_cycle": MAX_CYCLE, "status": "complete"}

                save_progress(progress)

                print(f"    Finished all cycles through {MAX_CYCLE}")

    print()
    print("=" * 80)
    print("SNL DOWNLOAD COMPLETE")
    print("=" * 80)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
DATASETS = {
    # "samsung_25r": fetch_samsung_25r,
    # "lg_mj1": fetch_lg_mj1,
    # "sony_vtc5a": fetch_sony_vtc5a,
    "sandia_snl": fetch_sandia_snl,
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
