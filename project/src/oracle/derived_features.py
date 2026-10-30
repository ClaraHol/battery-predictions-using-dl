"""Compute paper-derived stoichiometry features for experimental inference rows.

Place at project/src/oracle/derived_features.py and import from inferencers.
Requires physical estimates for theta_n_init, theta_p_init, eps_s_n, eps_s_p.
"""
import importlib

import numpy as np
import pandas as pd

FARADAY = 96485.33212  # As/mol
REQUIRED = (
    "theta_n_init_physical",
    "theta_p_init_physical",
    "eps_s_n_physical",
    "eps_s_p_physical",
)
DIRECTIONS = {"charge", "discharge"}


def chemistry_constants(chemistry):
    """Fixed battery properties from the chemistry's simulation parameter module."""
    module = importlib.import_module(
        f"src.simulation.parameters.parameter_{chemistry}"
    )
    # These names are defined in the project's parameter_*.py files.
    names = ("capa_nom", "L_n", "L_p", "area", "Cs_max_n", "Cs_max_p")
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{chemistry}: missing fixed constants {missing}")

    q_ah = float(module.capa_nom)
    ln, lp = float(module.L_n), float(module.L_p)
    area = float(module.area)
    cn, cp = float(module.Cs_max_n), float(module.Cs_max_p)

    vals = np.array([q_ah, ln, lp, area, cn, cp], dtype=float)
    if not np.isfinite(vals).all() or (vals <= 0).any():
        raise ValueError(f"{chemistry}: nonpositive/nonfinite battery constants")

    return q_ah * 3600.0, ln, lp, area, cn, cp


def add_derived_features(results):
    """Return a copy of the EFC0/EFC50 inference dataframe with 3 new columns.

    Computes derived features *after* the EFC50 lower/upper estimates have
    already been interpolated. The conversion depends on the direction of the
    reference-performance-test curve used to train the MLP:

    * ``charge`` models infer the fully discharged stoichiometries
      (theta_l_n/theta_l_p), so the fully charged stoichiometries are obtained
      by adding/subtracting the charge-capacity increment.
    * ``discharge`` models infer the fully charged stoichiometries
      (theta_h_n/theta_h_p), so the fully discharged stoichiometries are first
      obtained by subtracting/adding the discharge-capacity increment.

    The output is not clipped.
    """
    missing = [col for col in ("chemistry", "direction", *REQUIRED) if col not in results]
    if missing:
        raise ValueError(
            f"Cannot derive features; missing columns {missing}. "
            "The model must output physical stoichiometries and volume fractions, "
            "and the inference direction must be specified as 'charge' or 'discharge'."
        )

    df = results.copy()
    directions = df["direction"].astype(str).str.lower()
    invalid_directions = sorted(set(directions) - DIRECTIONS)
    if invalid_directions:
        raise ValueError(
            f"Cannot derive features; invalid inference direction(s) {invalid_directions}. "
            "Expected 'charge' or 'discharge'."
        )
    df["direction"] = directions

    for (chemistry, direction), indices in df.groupby(
        ["chemistry", "direction"], sort=False
    ).groups.items():
        q, ln, lp, area, cn, cp = chemistry_constants(chemistry)
        group = df.loc[indices]

        tn = group["theta_n_init_physical"].to_numpy(dtype=float)
        tp = group["theta_p_init_physical"].to_numpy(dtype=float)
        en = group["eps_s_n_physical"].to_numpy(dtype=float)
        ep = group["eps_s_p_physical"].to_numpy(dtype=float)

        # Note: nonphysical MLP outputs can cause nonsensical derived features.
        if not np.isfinite(np.column_stack([tn, tp, en, ep])).all():
            raise ValueError(f"{chemistry}: nonfinite inferred parameters")
        if (en <= 0).any() or (ep <= 0).any():
            raise ValueError(
                f"{chemistry}: inferred active material fractions must be > 0"
            )

        delta_n = q / (ln * area * cn * en * FARADAY)
        delta_p = q / (lp * area * cp * ep * FARADAY)

        if direction == "charge":
            # theta_n_init/theta_p_init are theta_l_n/theta_l_p.
            theta_l_n = tn
            theta_l_p = tp
            theta_h_n = theta_l_n + delta_n
            theta_h_p = theta_l_p - delta_p
        else:
            # theta_n_init/theta_p_init are theta_h_n/theta_h_p.
            theta_h_n = tn
            theta_h_p = tp
            theta_l_n = theta_h_n - delta_n
            theta_l_p = theta_h_p + delta_p

        df.loc[indices, "theta_h_n_physical"] = theta_h_n
        df.loc[indices, "theta_h_p_physical"] = theta_h_p
        df.loc[indices, "theta_off_physical"] = (
            theta_l_p + theta_l_n * (ln * cn * en) / (lp * cp * ep)
        )

    return df
