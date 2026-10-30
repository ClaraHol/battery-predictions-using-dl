#!/usr/bin/env python3
"""
Why is `checks` empty for a cell?  Walks the same stages as _capacity_check_curves and
reports where it stops, using ONE check-up file per cell.

    python debug_checks.py DATA_ROOT SUMMARY_DIR --log missing_log.txt [--out debug.csv]
    python debug_checks.py DATA_ROOT SUMMARY_DIR --cells 389 440 247
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tum_processing import load_fields, to_hours, I_THR
except ImportError:
    from build_cell_csvs import load_fields, to_hours, I_THR


def index_all(root):
    by_cell = defaultdict(lambda: {"cu": defaultdict(list), "cyc": 0, "loose": []})
    for f in Path(root).rglob("*.mat"):
        m_id = re.search(r"BW-VTC-(\d+)", f.name)
        if not m_id:
            continue
        c = int(m_id.group(1))
        m_cu = re.search(r"(?:^|[\\/])CU(\d+)(?:[\\/])", str(f), re.I)
        m_cy = re.search(r"(?:^|[\\/])Cyc(\d+)(?:[\\/])", str(f), re.I)
        if m_cu:
            by_cell[c]["cu"][int(m_cu.group(1))].append(f)
        elif m_cy:
            by_cell[c]["cyc"] += 1
        else:
            by_cell[c]["loose"].append(str(f))
    return by_cell


def inspect_file(path):
    out = {}
    fields = ["Time", "U", "I", "T1", "Command"]
    try:
        d = load_fields(path, fields)
    except Exception as e:                                   # noqa
        return {"stage": f"load error: {type(e).__name__}: {e}"}
    out["fields_loaded"] = ",".join(d)
    if "I" not in d or "U" not in d or "Time" not in d:
        return {**out, "stage": "Time/U/I missing in file"}
    I, U = d["I"], d["U"]
    out.update(rows=len(I), I_min=round(float(np.nanmin(I)), 3), I_max=round(float(np.nanmax(I)), 3),
               U_min=round(float(np.nanmin(U)), 3), U_max=round(float(np.nanmax(U)), 3))
    if "Command" in d:
        cmd = np.asarray(d["Command"])
        vals = pd.unique(pd.Series(cmd).astype(str).str.strip())
        out["Command_values"] = ",".join(map(str, vals[:6]))
        n_dis = int(sum(str(x).strip().lower() == "discharge" for x in cmd[:200000]))   # first 200k rows
        out["Command_discharge_rows_first200k"] = n_dis
        out["len_Command_eq_len_I"] = len(cmd) == len(I)
    dis = I < -I_THR
    edges = np.flatnonzero(np.diff(np.r_[False, dis, False].astype(int)))
    s, e = edges[::2], edges[1::2]
    out["n_negative_current_blocks"] = len(s)
    out["n_cccv_blocks(U>4.1->U<2.7)"] = int(sum(1 for a, b in zip(s, e) if b - a >= 2 and U[a] > 4.1 and U[b - 1] < 2.7))
    if "Command_discharge_rows_first200k" in out and out["Command_discharge_rows_first200k"] == 0 and len(s):
        out["stage"] = "Command loaded but never equals 'Discharge' (see Command_values) -> Command-based mask finds nothing"
    elif len(s) == 0:
        out["stage"] = "no negative-current block (check I sign / units)"
    else:
        out["stage"] = "OK: curve extraction works on this file"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("summary_dir")
    ap.add_argument("--log"); ap.add_argument("--cells", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="debug_checks.csv")
    a = ap.parse_args()

    cells = list(a.cells)
    print(cells)
    if a.log:
        cells += [int(m) for m in dict.fromkeys(re.findall(r"BW-VTC-(\d+)\]? - Missing", Path(a.log).read_text()))]
    cells = sorted(set(cells))
    idx = index_all(a.root)
    rows = []
    for c in cells:
        info = idx.get(c)
        r = {"cell": c}
        sp = Path(a.summary_dir) / f"BW-VTC-{c:03d}.csv"
        s = pd.read_csv(sp) if sp.exists() else None
        r["summary_exists"] = s is not None
        r["summary_CU"] = "" if s is None else f"{int(s.CU.min())}..{int(s.CU.max())} ({len(s)})"
        if info is None:
            r["stage"] = "NO .mat file with BW-VTC-%03d in its name under root" % c
        elif not info["cu"]:
            ex = (info["loose"] or ["-"])[0]
            r["stage"] = (f"cell has files but none inside a CU### folder (cyc files: {info['cyc']}, "
                          f"loose files: {len(info['loose'])}, e.g. {ex})")
        else:
            cu_nums = sorted(info["cu"])
            r["idx_CU"] = f"{cu_nums[0]}..{cu_nums[-1]} ({len(cu_nums)})"
            if s is None:
                r["stage"] = "no summary CSV for this cell"
            else:
                sv = s.dropna(subset=["EFC"]) if "EFC" in s else s.iloc[0:0]
                common = sorted(set(cu_nums) & set(sv.CU.astype(int)))
                if sv.empty:
                    r["stage"] = "summary has no valid EFC (old summary version?)"
                elif not common:
                    r["stage"] = "CU numbers in files and in summary do not overlap"
                else:
                    r["test_CU"] = common[0]
                    r.update(inspect_file(info["cu"][common[0]][0]))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df["stage"].str.slice(0, 90).value_counts().to_string())
    print(f"\nFull table: {a.out}")
    show = [c for c in ("cell", "summary_CU", "idx_CU", "fields_loaded", "Command_values", "n_negative_current_blocks",
                        "n_cccv_blocks(U>4.1->U<2.7)", "stage") if c in df]
    print(df[show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()