import numpy as np
from tum_processing import load_fields

def diagnose_time_units(path):
    d = load_fields(path, ["Time", "I"])

    t = np.asarray(d["Time"], dtype=float)
    I = np.asarray(d["I"], dtype=float)

    dt = np.diff(t)
    positive_dt = dt[dt > 0]

    if len(positive_dt) == 0:
        return

    median_dt = np.median(positive_dt)

    # Current held until next measurement
    charge_raw = np.sum(
        np.maximum(I[:-1], 0) *
        np.clip(dt, 0, None)
    )

    # Interpretation A: raw Time is in hours
    ah_if_hours = charge_raw

    # Interpretation B: raw Time is in seconds
    ah_if_seconds = charge_raw / 3600

    print(f"\nFile: {path}")
    print(f"Median raw dt: {median_dt:.8f}")
    print(f"Charge if hours:   {ah_if_hours:.4f} Ah")
    print(f"Charge if seconds: {ah_if_seconds:.4f} Ah")
    print(f"Current range: {I.min():.4f} to {I.max():.4f} A")

