"""Extract experimental EFC0/EFC50 features with chemistry-specific MLPs.

The processed experimental data are expected in the long-form Parquet schema:

Curves (one row per measured point):
    cell_id, chemistry, curve_type, time_s, voltage_V, current_A, rpt_efc

Metadata (one row per cell) must also contain the downstream target/conditions:
    cycle_life_efc, cycle_life_censored, cycle_life_soh_threshold,
    temperature_C, charge_c_rate, discharge_c_rate

Only cells that actually reached the requested SOH threshold are processed.
Censored cells are excluded before MLP inference. EFC50 features are obtained by
running the MLP separately on the bracketing RPT curves and interpolating the
inferred normalized features with the metadata weights. VTC5A, VTC6, Samsung-MJ1 (Zhu Dataset 2), LG-MJ1 (VITO/EVERLASTING),
and Samsung-25R (Zhu Dataset 3) are currently supported. Multiple raw datasets may
share one chemistry/model; provenance is retained separately in the output.

For downstream physics features, parameters trained with log-uniform priors are
saved on the same base-10 logarithmic scale used in the paper. Thus D_s_n,
D_s_p, k_n, k_p and R_f are represented as log10(physical value), while the
remaining parameters stay on their native physical scale. Raw physical values
are retained in separate *_physical columns for auditability.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# This module is expected to live next to this script, as in the original code.
PROCESSED_DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data")
MODEL_DIR = PROJECT_ROOT / "models" / "MLP_models"
OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "inference_results"
    / "experimental_inference_mlp2.csv"
)
DEVICE = torch.device("cpu")

EFC50_TARGET = 50.0
LIFETIME_SOH_THRESHOLD = 0.90
EXPECTED_CURVE_TYPES = ("efc0", "efc50_lower", "efc50_upper")

# The paper reports these five physics features on a base-10 logarithmic scale.
# Parameter modules may include units in their names, so validation below compares
# the portion before the first `` [``.
EXPECTED_LOG10_PARAMETERS = {"D_s_n", "D_s_p", "k_n", "k_p", "R_f"}

# Chemistry: rated capacity (Ah), reference current magnitude (A), voltage limits (V)
# Keep these identical to the training-time observation normalization.
CHEMISTRY_CONFIG = {
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6": (3.0, 0.6, 2.5, 4.25),
    # Zhu2022 Dataset_2_NCM_battery: 3.5 Ah MJ1, 1C discharge, 2.5--4.2 V.
    # This uses the same MJ1 chemistry/model that will later also be used for
    # the separately processed LG-MJ1/VITO dataset.
    "MJ1": (3.5, 3.5, 2.5, 4.20),
    # Zhu2022 Dataset_3_NCM_NCA_battery: 2.5 Ah Samsung 25R.
    # Discovery Learning uses the 0.5C constant-current charge profile.
    "25R": (2.5, 1.25, 2.5, 4.20),
}

CHEMISTRY_DIRECTION = {
    "VTC5A": "discharge",
    "VTC6": "discharge",
    "MJ1": "discharge",
    "25R": "charge",
}

# Inputs are keyed by source, not chemistry. This deliberately permits multiple
# independently processed datasets to map to the same final chemistry/model.
DATA_SOURCES = {
    "vtc5a": {
        "chemistry": "VTC5A",
        "curves": PROCESSED_DATA_DIR / "vtc5a" / "vtc5a_all_curves.parquet",
        "metadata": PROCESSED_DATA_DIR / "vtc5a" / "vtc5a_all_metadata.parquet",
    },
    "vtc6": {
        "chemistry": "VTC6",
        "curves": PROCESSED_DATA_DIR / "vtc6" / "vtc6_all_curves.parquet",
        "metadata": PROCESSED_DATA_DIR / "vtc6" / "vtc6_all_metadata.parquet",
    },
    "samsung_mj1": {
        "chemistry": "MJ1",
        "curves": (
            PROCESSED_DATA_DIR
            / "samsung_mj1"
            / "samsung_mj1_all_curves.parquet"
        ),
        "metadata": (
            PROCESSED_DATA_DIR
            / "samsung_mj1"
            / "samsung_mj1_all_metadata.parquet"
        ),
    },
    "lg_mj1": {
        "chemistry": "MJ1",
        "curves": (
            PROCESSED_DATA_DIR
            / "lg_mj1"
            / "lg_mj1_all_curves.parquet"
        ),
        "metadata": (
            PROCESSED_DATA_DIR
            / "lg_mj1"
            / "lg_mj1_all_metadata.parquet"
        ),
    },
    "samsung_25r": {
        "chemistry": "25R",
        "curves": (
            PROCESSED_DATA_DIR
            / "samsung_25r"
            / "samsung_25r_all_curves.parquet"
        ),
        "metadata": (
            PROCESSED_DATA_DIR
            / "samsung_25r"
            / "samsung_25r_all_metadata.parquet"
        ),
    },
}

CURVE_COLUMNS = [
    "cell_id",
    "chemistry",
    "curve_type",
    "time_s",
    "voltage_V",
    "current_A",
    "rpt_efc",
]

BASE_METADATA_COLUMNS = [
    "cell_id",
    "chemistry",
    "efc0",
    "efc50_lower",
    "efc50_upper",
    "weight_lower",
    "weight_upper",
    "direction",
]

TARGET_METADATA_COLUMNS = [
    "cycle_life_efc",
    "cycle_life_censored",
    "cycle_life_soh_threshold",
    "temperature_C",
    "charge_c_rate",
    "discharge_c_rate",
]

# These are copied to the output when present, but are not needed for inference.
OPTIONAL_OUTPUT_METADATA_COLUMNS = [
    "lifetime_status",
    "lifetime_censor_reason",
    "temperature_C_raw",
    "ambient_temperature_raw_C",
    "charge_c_rate_raw",
    "discharge_c_rate_raw",
    "charge_voltage_V",
    "charge_cc_c_rate_protocol",
    "source_family",
    "source_dataset",
    "dataset_provenance",
    "source_file",
    "condition",
    "cell_tag",
    "cell_number",
    "test_type",
    "soc_lower_pct",
    "soc_upper_pct",
    "c_rate_method",
    "n_rpts_single_detected_cycle",
    "n_rpts_nonpair",
]


# Exactly the same residual network as in the supplied train_mlp.py.
class ResidualBlock(nn.Module):
    def __init__(self, hidden_features, dropout_probability, use_batch_norm):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn0 = nn.BatchNorm1d(hidden_features, eps=1e-3)
            self.bn1 = nn.BatchNorm1d(hidden_features, eps=1e-3)
        self.linear0 = nn.Linear(hidden_features, hidden_features)
        self.linear1 = nn.Linear(hidden_features, hidden_features)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.bn0(temps)
        temps = self.activation(temps)
        temps = self.linear0(temps)
        if self.use_batch_norm:
            temps = self.bn1(temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear1(temps)
        return inputs + temps


class ResidualNetRegressor(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        num_blocks,
        dropout_probability,
        use_batch_norm,
    ):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_features,
                    dropout_probability,
                    use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.initial_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.final_layer(h)


def load_mlps():
    """Load the two required chemistry-specific MLPs and fail loudly if absent."""
    models = {}

    for chemistry in CHEMISTRY_CONFIG:
        path = MODEL_DIR / chemistry / f"mlp_{chemistry}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required MLP for {chemistry}: {path}")

        module = importlib.import_module(
            f"src.simulation.parameters.parameter_{chemistry}"
        )
        names = list(module.params_log.keys())
        flags = list(module.params_log_flag.values())

        log_names = {
            name.split(" [", 1)[0]
            for name, flag in zip(names, flags)
            if flag
        }
        if log_names != EXPECTED_LOG10_PARAMETERS:
            raise ValueError(
                f"{chemistry} log-parameter flags do not match the paper: "
                f"found {sorted(log_names)}, expected "
                f"{sorted(EXPECTED_LOG10_PARAMETERS)}"
            )

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)

        module_low = torch.tensor(
            [limits[0] for limits in module.params_log.values()],
            dtype=torch.float32,
        )
        module_high = torch.tensor(
            [limits[1] for limits in module.params_log.values()],
            dtype=torch.float32,
        )
        low = torch.as_tensor(
            checkpoint.get("prior_min", module_low),
            dtype=torch.float32,
        )
        high = torch.as_tensor(
            checkpoint.get("prior_max", module_high),
            dtype=torch.float32,
        )

        saved_names = checkpoint.get("parameter_names", names)
        if list(saved_names) != names:
            raise ValueError(f"Parameter ordering mismatch for {chemistry}")
        if len(names) != 11 or not torch.all(high > low):
            raise ValueError(f"Invalid parameters/prior for {chemistry}")

        net = ResidualNetRegressor(
            in_features=100,
            out_features=len(names),
            hidden_features=256,
            num_blocks=9,
            dropout_probability=0.2,
            use_batch_norm=True,
        ).to(DEVICE)
        net.load_state_dict(state, strict=True)
        net.eval()

        models[chemistry] = {
            "net": net,
            "param_names": names,
            "log_flags": flags,
            "prior_min": low,
            "prior_range": high - low,
        }
        print(f"Loaded MLP: {chemistry}")

    return models


def _require_columns(df, required, label):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_dataset(curves, metadata, chemistry):
    """Validate the new curve/metadata contract before any MLP inference."""
    _require_columns(curves, CURVE_COLUMNS, f"{chemistry} curves")
    _require_columns(
        metadata,
        BASE_METADATA_COLUMNS + TARGET_METADATA_COLUMNS,
        f"{chemistry} metadata",
    )

    if curves.empty:
        raise ValueError(f"{chemistry} curves are empty")
    if metadata.empty:
        raise ValueError(f"{chemistry} metadata are empty")

    curve_chemistries = set(curves["chemistry"].dropna().astype(str).unique())
    metadata_chemistries = set(metadata["chemistry"].dropna().astype(str).unique())
    if curve_chemistries != {chemistry}:
        raise ValueError(
            f"{chemistry} curve file contains chemistry values {curve_chemistries}"
        )
    if metadata_chemistries != {chemistry}:
        raise ValueError(
            f"{chemistry} metadata file contains chemistry values "
            f"{metadata_chemistries}"
        )

    if metadata["cell_id"].duplicated().any():
        duplicated = metadata.loc[
            metadata["cell_id"].duplicated(keep=False), "cell_id"
        ].unique()
        raise ValueError(
            f"{chemistry} metadata contains duplicate cell_id rows: "
            f"{duplicated[:10].tolist()}"
        )

    directions = set(metadata["direction"].dropna().astype(str).str.lower().unique())
    expected_direction = CHEMISTRY_DIRECTION[chemistry]
    if directions != {expected_direction}:
        raise ValueError(
            f"{chemistry} must use {expected_direction} reference curves, "
            f"found directions {directions}"
        )

    metadata_ids = set(metadata["cell_id"].astype(str))
    curve_ids = set(curves["cell_id"].astype(str))
    if metadata_ids != curve_ids:
        only_metadata = sorted(metadata_ids - curve_ids)
        only_curves = sorted(curve_ids - metadata_ids)
        raise ValueError(
            f"{chemistry} cell IDs differ between metadata and curves. "
            f"metadata-only={only_metadata[:10]}, curves-only={only_curves[:10]}"
        )

    actual_curve_types = set(curves["curve_type"].dropna().astype(str).unique())
    expected_curve_types = set(EXPECTED_CURVE_TYPES)
    if actual_curve_types != expected_curve_types:
        raise ValueError(
            f"{chemistry} curve types are {sorted(actual_curve_types)}, expected "
            f"{sorted(expected_curve_types)}"
        )

    counts = (
        curves.groupby(["cell_id", "curve_type"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=EXPECTED_CURVE_TYPES, fill_value=0)
    )
    missing_curves = counts.index[(counts == 0).any(axis=1)].tolist()
    if missing_curves:
        raise ValueError(
            f"{chemistry} cells missing one or more required curves: "
            f"{missing_curves[:10]}"
        )

    numeric_meta = [
        "efc0",
        "efc50_lower",
        "efc50_upper",
        "weight_lower",
        "weight_upper",
    ]
    if not np.isfinite(metadata[numeric_meta].to_numpy(dtype=float)).all():
        raise ValueError(f"{chemistry} metadata contains NaN/Inf in EFC or weight fields")

    if not np.allclose(metadata["efc0"].to_numpy(dtype=float), 0.0, atol=1e-8):
        raise ValueError(f"{chemistry} metadata efc0 must be 0")

    lower = metadata["efc50_lower"].to_numpy(dtype=float)
    upper = metadata["efc50_upper"].to_numpy(dtype=float)
    weight_lower = metadata["weight_lower"].to_numpy(dtype=float)
    weight_upper = metadata["weight_upper"].to_numpy(dtype=float)

    if np.any(upper <= lower):
        raise ValueError(f"{chemistry} has efc50_upper <= efc50_lower")
    if np.any(lower > EFC50_TARGET) or np.any(upper < EFC50_TARGET):
        raise ValueError(f"{chemistry} has an EFC50 pair that does not bracket 50")
    if np.any((weight_lower < 0) | (weight_lower > 1)):
        raise ValueError(f"{chemistry} weight_lower is outside [0, 1]")
    if np.any((weight_upper < 0) | (weight_upper > 1)):
        raise ValueError(f"{chemistry} weight_upper is outside [0, 1]")
    if not np.allclose(weight_lower + weight_upper, 1.0, atol=1e-8):
        raise ValueError(f"{chemistry} interpolation weights do not sum to 1")

    expected_upper_weight = (EFC50_TARGET - lower) / (upper - lower)
    if not np.allclose(weight_upper, expected_upper_weight, atol=1e-7, rtol=1e-7):
        max_error = np.max(np.abs(weight_upper - expected_upper_weight))
        raise ValueError(
            f"{chemistry} weight_upper does not match the EFC bracket formula; "
            f"max error={max_error:.3e}"
        )

    # Every stored curve should have one constant rpt_efc value, and that value
    # should agree with the per-cell metadata.
    curve_efc = (
        curves.groupby(["cell_id", "curve_type"], sort=False)["rpt_efc"]
        .agg(["nunique", "first"])
        .reset_index()
    )
    if (curve_efc["nunique"] != 1).any():
        bad = curve_efc.loc[curve_efc["nunique"] != 1, ["cell_id", "curve_type"]]
        raise ValueError(
            f"{chemistry} has curves with non-constant rpt_efc: "
            f"{bad.head(10).to_dict('records')}"
        )

    expected_efc_columns = {
        "efc0": "efc0",
        "efc50_lower": "efc50_lower",
        "efc50_upper": "efc50_upper",
    }
    metadata_indexed = metadata.set_index("cell_id")
    for curve_type, metadata_column in expected_efc_columns.items():
        observed = curve_efc.loc[curve_efc["curve_type"] == curve_type].set_index(
            "cell_id"
        )["first"]
        expected = metadata_indexed.loc[observed.index, metadata_column]
        if not np.allclose(
            observed.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            atol=1e-8,
            rtol=1e-8,
        ):
            raise ValueError(
                f"{chemistry} {curve_type} rpt_efc values do not match metadata"
            )

    # Check that the stored current magnitude is consistent with the model's
    # reference current. Median is used so a single boundary/rest sample does
    # not cause a false failure.
    expected_current = CHEMISTRY_CONFIG[chemistry][1]
    current_medians = curves.groupby(["cell_id", "curve_type"])["current_A"].median()
    if not np.isfinite(current_medians.to_numpy(dtype=float)).all():
        raise ValueError(f"{chemistry} contains invalid current values")
    if not np.allclose(
        current_medians.to_numpy(dtype=float),
        expected_current,
        rtol=0.15,
        atol=0.05,
    ):
        bad = current_medians[
            ~np.isclose(
                current_medians.to_numpy(dtype=float),
                expected_current,
                rtol=0.15,
                atol=0.05,
            )
        ]
        raise ValueError(
            f"{chemistry} curve currents do not match the expected "
            f"{expected_current:g} A {CHEMISTRY_DIRECTION[chemistry]} reference current. "
            f"Examples: {bad.head().to_dict()}"
        )

    print(
        f"Validated {chemistry}: {len(metadata)} cells, {len(curves):,} curve points, "
        "all three RPT curve types present"
    )


def _as_bool_series(series, label):
    """Return a strict boolean Series; reject ambiguous object/string values."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapped = series.map(
        lambda x: x
        if isinstance(x, (bool, np.bool_))
        else {"true": True, "false": False}.get(str(x).strip().lower(), np.nan)
    )
    if mapped.isna().any():
        bad = series.loc[mapped.isna()].astype(str).unique()[:10].tolist()
        raise ValueError(f"{label} contains non-boolean values: {bad}")
    return mapped.astype(bool)


def select_observed_lifetimes(curves, metadata, chemistry):
    """Keep only cells with an observed (not censored) lifetime at 90% SOH."""
    censored = _as_bool_series(
        metadata["cycle_life_censored"],
        f"{chemistry} cycle_life_censored",
    )
    lifetime = pd.to_numeric(metadata["cycle_life_efc"], errors="coerce")
    threshold = pd.to_numeric(
        metadata["cycle_life_soh_threshold"], errors="coerce"
    )

    finite_threshold = np.isfinite(threshold.to_numpy(dtype=float))
    if not finite_threshold.all():
        raise ValueError(f"{chemistry} has missing/non-finite SOH thresholds")
    if not np.allclose(
        threshold.to_numpy(dtype=float),
        LIFETIME_SOH_THRESHOLD,
        atol=1e-8,
        rtol=0.0,
    ):
        found = sorted(threshold.unique().tolist())
        raise ValueError(
            f"{chemistry} lifetime threshold differs from requested "
            f"{LIFETIME_SOH_THRESHOLD:.0%}: {found}"
        )

    observed = (~censored) & np.isfinite(lifetime.to_numpy(dtype=float))
    observed &= lifetime.to_numpy(dtype=float) > 0.0

    selected_metadata = metadata.loc[observed].copy()
    selected_ids = set(selected_metadata["cell_id"].astype(str))
    selected_curves = curves.loc[
        curves["cell_id"].astype(str).isin(selected_ids)
    ].copy()

    print(
        f"{chemistry}: using {len(selected_metadata)}/{len(metadata)} cells with "
        f"observed {LIFETIME_SOH_THRESHOLD:.0%} SOH lifetime; "
        f"excluded {int(censored.sum())} censored cells"
    )

    if selected_metadata.empty:
        raise ValueError(
            f"{chemistry} has no cells with an observed "
            f"{LIFETIME_SOH_THRESHOLD:.0%} SOH lifetime"
        )

    # Conditions are not an inference requirement, but downstream lifetime models
    # need them. Keep the cells and report missing values explicitly rather than
    # silently introducing a second selection criterion.
    condition_columns = ["temperature_C", "charge_c_rate", "discharge_c_rate"]
    missing_conditions = selected_metadata[condition_columns].isna().any(axis=1)
    if missing_conditions.any():
        bad_ids = selected_metadata.loc[
            missing_conditions, "cell_id"
        ].astype(str).tolist()
        print(
            f"WARNING: {chemistry} has {len(bad_ids)} observed-lifetime cells "
            f"with missing condition metadata: {bad_ids[:10]}"
        )

    return selected_curves, selected_metadata


def load_experimental_data():
    """Load each processed source independently, then map it to a chemistry model."""
    datasets = {}

    for source_name, files in DATA_SOURCES.items():
        chemistry = files["chemistry"]
        for label in ("curves", "metadata"):
            path = files[label]
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {source_name} ({chemistry}) {label} file: {path}"
                )

        curves = pd.read_parquet(files["curves"], columns=CURVE_COLUMNS)
        # Metadata schemas differ by source, so read the full (small) metadata
        # table and require only the common inference/target contract.
        metadata = pd.read_parquet(files["metadata"])
        validate_dataset(curves, metadata, chemistry)
        curves, metadata = select_observed_lifetimes(
            curves, metadata, chemistry
        )
        datasets[source_name] = {
            "chemistry": chemistry,
            "curves": curves,
            "metadata": metadata,
        }
        print(
            f"Loaded source {source_name}: chemistry={chemistry}, "
            f"observed-lifetime cells={len(metadata)}"
        )

    return datasets


def make_observation(voltage, time, chemistry):
    """Apply the same 100-dimensional observation preprocessing as training."""
    voltage = np.asarray(voltage, dtype=float)
    time = np.asarray(time, dtype=float)

    mask = np.isfinite(voltage) & np.isfinite(time)
    voltage = voltage[mask]
    time = time[mask]

    order = np.argsort(time)
    voltage = voltage[order]
    time = time[order]

    time, unique_idx = np.unique(time, return_index=True)
    voltage = voltage[unique_idx]

    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("curve has insufficient valid time points")

    capacity, current, v_min, v_max = CHEMISTRY_CONFIG[chemistry]
    time = time - time[0]
    t_interp = np.linspace(0.0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)

    t_typ = abs(3600.0 * capacity / current)
    observation = np.sqrt(
        (t_interp / t_typ) ** 2
        + ((v_interp - v_min) / (v_max - v_min)) ** 2
    )

    if not np.isfinite(observation).all():
        raise ValueError("observation preprocessing produced NaN or Inf")

    return observation


def predict_normalized(model, voltage, time, chemistry):
    observation = make_observation(voltage, time, chemistry)
    x = torch.tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        normalized = model["net"](x).squeeze(0).cpu()

    if not torch.isfinite(normalized).all():
        raise ValueError("MLP output contains NaN or Inf")

    return normalized.numpy()


def normalized_to_parameter_scales(model, normalized):
    """Return prior-space, physical, and paper-style feature values.

    The model is trained against ``theta`` normalized to chemistry-specific
    prior bounds. For parameters with ``log_flags=True``, theta is log10 of the
    physical parameter. The paper uses those five parameters on this log10
    scale for downstream lifetime prediction. Non-log parameters are already
    on their physical scale.
    """
    normalized = torch.as_tensor(normalized, dtype=torch.float32)
    theta = normalized * model["prior_range"] + model["prior_min"]

    physical = torch.stack(
        [
            10.0 ** theta[j] if flag else theta[j]
            for j, flag in enumerate(model["log_flags"])
        ]
    )

    # For log-scaled parameters, theta itself is log10(physical). For all
    # others, theta equals the physical value. This is the representation used
    # as the downstream physics feature.
    feature = theta.clone()

    return theta.numpy(), physical.numpy(), feature.numpy()


def result_row(
    cell_id, chemistry, curve_type, model, normalized, metadata_row
):
    _, physical, feature = normalized_to_parameter_scales(model, normalized)
    row = {
        "cell_id": cell_id,
        "chemistry": chemistry,
        "curve_type": curve_type,
        # Downstream target and cycling-condition features. These values are
        # cell-level metadata and are therefore repeated on EFC0/EFC50 rows.
        "cycle_life_efc": float(metadata_row.cycle_life_efc),
        "cycle_life_censored": bool(metadata_row.cycle_life_censored),
        "cycle_life_soh_threshold": float(
            metadata_row.cycle_life_soh_threshold
        ),
        "temperature_C": float(metadata_row.temperature_C)
        if pd.notna(metadata_row.temperature_C)
        else np.nan,
        "charge_c_rate": float(metadata_row.charge_c_rate)
        if pd.notna(metadata_row.charge_c_rate)
        else np.nan,
        "discharge_c_rate": float(metadata_row.discharge_c_rate)
        if pd.notna(metadata_row.discharge_c_rate)
        else np.nan,
    }

    # Preserve useful chemistry-specific audit/protocol metadata when available.
    row_dict = metadata_row._asdict()
    for column in OPTIONAL_OUTPUT_METADATA_COLUMNS:
        if column in row_dict:
            row[column] = row_dict[column]

    for name, norm, phys, feature_value in zip(
        model["param_names"],
        normalized,
        physical,
        feature,
    ):
        row[f"{name}_normalized"] = float(norm)
        row[f"{name}_physical"] = float(phys)

        # ``*_feature`` is the representation intended for downstream models:
        # log10 for D_s_n, D_s_p, k_n, k_p and R_f; native scale otherwise.
        row[f"{name}_feature"] = float(feature_value)

    return row


def infer_curve(model, curve_data, chemistry):
    return predict_normalized(
        model,
        curve_data["voltage_V"].to_numpy(),
        curve_data["time_s"].to_numpy(),
        chemistry,
    )


def run_chemistry_inference(chemistry, curves, metadata, model):
    """Infer EFC0 and interpolated EFC50 features for one chemistry."""
    rows = []
    failures = []

    grouped_curves = {
        (cell_id, curve_type): group
        for (cell_id, curve_type), group in curves.groupby(
            ["cell_id", "curve_type"], sort=False
        )
    }

    for meta in metadata.itertuples(index=False):
        cell_id = meta.cell_id
        try:
            efc0_curve = grouped_curves[(cell_id, "efc0")]
            lower_curve = grouped_curves[(cell_id, "efc50_lower")]
            upper_curve = grouped_curves[(cell_id, "efc50_upper")]

            efc0_normalized = infer_curve(model, efc0_curve, chemistry)
            lower_normalized = infer_curve(model, lower_curve, chemistry)
            upper_normalized = infer_curve(model, upper_curve, chemistry)

            # Interpolate the inferred feature representation, not the raw curves.
            # This is also consistent for log-scaled parameters because normalized
            # outputs are affine in the model's theta/prior coordinates.
            efc50_normalized = (
                float(meta.weight_lower) * lower_normalized
                + float(meta.weight_upper) * upper_normalized
            )

            rows.append(
                result_row(
                    cell_id,
                    chemistry,
                    "efc0",
                    model,
                    efc0_normalized,
                    meta,
                )
            )
            rows.append(
                result_row(
                    cell_id,
                    chemistry,
                    "efc50",
                    model,
                    efc50_normalized,
                    meta,
                )
            )
        except (KeyError, ValueError) as exc:
            failures.append((cell_id, str(exc)))
            print(f"Skipping {chemistry} {cell_id}: {exc}")

    result = pd.DataFrame(rows)
    return result, failures




def add_derived_features(results):
    """Calculate the three dependent stoichiometry features locally.

    The 11 inferred parameters contain the two endpoint stoichiometries at the
    RPT cutoff.  Their interpretation depends on the RPT direction:

    * charge RPTs (25R): theta_n_init/theta_p_init are theta_l^n/theta_l^p;
      derive theta_h^n/theta_h^p.
    * discharge RPTs (VTC5A, VTC6, MJ1): theta_n_init/theta_p_init are
      theta_h^n/theta_h^p; derive theta_l^n/theta_l^p.

    The equations are the ones given in Supplementary Information, equations
    (14)-(19).  The rated capacity from CHEMISTRY_CONFIG is used here to retain
    the same capacity convention as the existing extractor.

    The function deliberately lives in this extractor rather than importing a
    shared helper so the charge/discharge interpretation is explicit and easy
    to audit alongside the RPT selection.
    """
    required = {
        "chemistry",
        "theta_n_init_physical",
        "theta_p_init_physical",
        "eps_s_n_physical",
        "eps_s_p_physical",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            f"Cannot calculate derived features; missing columns: {missing}"
        )

    results = results.copy()
    F_DEFAULT = 96485.0

    # Canonical names used by the paper's 14-feature representation.
    for column in (
        "theta_h_n_physical",
        "theta_l_n_physical",
        "theta_h_p_physical",
        "theta_l_p_physical",
        "theta_off_physical",
    ):
        results[column] = np.nan

    for chemistry in results["chemistry"].dropna().unique():
        mask = results["chemistry"] == chemistry
        direction = CHEMISTRY_DIRECTION[chemistry]

        module = importlib.import_module(
            f"src.simulation.parameters.parameter_{chemistry}"
        )

        def get_parameter(name, *aliases):
            for candidate in (name, *aliases):
                if hasattr(module, candidate):
                    return float(getattr(module, candidate))
            raise AttributeError(
                f"{chemistry}: parameter module has none of {name!r}, "
                f"{aliases!r}"
            )

        # These are fixed design/material quantities from the chemistry's
        # parameter module.  The active-material fractions are inferred values
        # and therefore come from each row below.
        area = get_parameter("area", "electrode_area")
        L_n = get_parameter("L_n", "L_negative")
        L_p = get_parameter("L_p", "L_positive")
        Cs_max_n = get_parameter("Cs_max_n", "c_s_max_n")
        Cs_max_p = get_parameter("Cs_max_p", "c_s_max_p")
        F = get_parameter("F") if hasattr(module, "F") else F_DEFAULT

        # Preserve the convention used by the current v2 extractor.  The
        # capacity is in Ah and the SI equations require As.
        Q_As = CHEMISTRY_CONFIG[chemistry][0] * 3600.0

        en = results.loc[mask, "eps_s_n_physical"].to_numpy(dtype=float)
        ep = results.loc[mask, "eps_s_p_physical"].to_numpy(dtype=float)
        theta_n = results.loc[mask, "theta_n_init_physical"].to_numpy(dtype=float)
        theta_p = results.loc[mask, "theta_p_init_physical"].to_numpy(dtype=float)

        if np.any(~np.isfinite(en)) or np.any(~np.isfinite(ep)):
            raise ValueError(f"{chemistry}: non-finite active-material fractions")
        if np.any(en <= 0) or np.any(ep <= 0):
            raise ValueError(f"{chemistry}: active-material fractions must be > 0")

        delta_n = Q_As / (L_n * area * Cs_max_n * en * F)
        delta_p = Q_As / (L_p * area * Cs_max_p * ep * F)

        if direction == "charge":
            # Equations (14)-(16): inferred endpoints are theta_l.
            theta_l_n = theta_n
            theta_l_p = theta_p
            theta_h_n = theta_l_n + delta_n
            theta_h_p = theta_l_p - delta_p
        elif direction == "discharge":
            # Equations (17)-(19): inferred endpoints are theta_h.
            theta_h_n = theta_n
            theta_h_p = theta_p
            theta_l_n = theta_h_n - delta_n
            theta_l_p = theta_h_p + delta_p
        else:
            raise ValueError(f"Unsupported RPT direction for {chemistry}: {direction}")

        # Equation (16)/(19).  This quantity is independent of the direction
        # once the corresponding lower-cutoff stoichiometries are known.
        theta_off = theta_l_p + theta_l_n * (
            L_n * Cs_max_n * en / (L_p * Cs_max_p * ep)
        )

        results.loc[mask, "theta_h_n_physical"] = theta_h_n
        results.loc[mask, "theta_l_n_physical"] = theta_l_n
        results.loc[mask, "theta_h_p_physical"] = theta_h_p
        results.loc[mask, "theta_l_p_physical"] = theta_l_p
        results.loc[mask, "theta_off_physical"] = theta_off

    # Add canonical paper-feature aliases.  The original 11-model-output
    # columns (theta_n_init_feature/theta_p_init_feature) are retained as
    # audit columns; downstream lifetime modelling should use the explicit
    # 14-column canonical set: 9 independent parameters plus the five
    # stoichiometry/offset features below.
    for name in (
        "theta_h_n",
        "theta_l_n",
        "theta_h_p",
        "theta_l_p",
        "theta_off",
    ):
        results[f"{name}_feature"] = results[f"{name}_physical"]

    return results


def print_normalized_means(results):
    """Print mean normalized MLP features for VTC5A and VTC6."""
    normalized_columns = [
        column for column in results.columns if column.endswith("_normalized")
    ]
    if not normalized_columns:
        print("No normalized feature columns available to summarize")
        return

    print("\nMean normalized features by battery type:")
    chemistry_means = (
        results.groupby("chemistry", sort=False)[normalized_columns]
        .mean()
        .reindex(CHEMISTRY_CONFIG.keys())
        .T
    )
    print(chemistry_means.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nMean normalized features by battery type and stage:")
    stage_means = (
        results.groupby(["chemistry", "curve_type"], sort=False)[normalized_columns]
        .mean()
        .T
    )
    print(stage_means.to_string(float_format=lambda x: f"{x:.6f}"))

    outside = (results[normalized_columns] < 0.0) | (
        results[normalized_columns] > 1.0
    )
    if outside.any().any():
        print("\nFraction of normalized predictions outside [0, 1]:")
        outside_by_chemistry = outside.copy()
        outside_by_chemistry.insert(0, "chemistry", results["chemistry"].to_numpy())
        outside_fraction = (
            outside_by_chemistry.groupby("chemistry", sort=False)[normalized_columns]
            .mean()
            .reindex(CHEMISTRY_CONFIG.keys())
            .T
        )
        print(outside_fraction.to_string(float_format=lambda x: f"{x:.3%}"))



def print_target_condition_summary(results):
    """Summarize saved lifetime and condition metadata once per cell."""
    cell_rows = (
        results.sort_values(["chemistry", "cell_id", "curve_type"])
        .drop_duplicates(["chemistry", "cell_id"])
        .copy()
    )

    print("\nObserved lifetime / condition summary:")
    for chemistry in CHEMISTRY_CONFIG:
        subset = cell_rows.loc[cell_rows["chemistry"] == chemistry]
        if subset.empty:
            continue
        life = subset["cycle_life_efc"].to_numpy(dtype=float)
        print(
            f"{chemistry}: n={len(subset)}, "
            f"lifetime mean={np.mean(life):.1f} EFC, "
            f"median={np.median(life):.1f} EFC, "
            f"range={np.min(life):.1f}..{np.max(life):.1f} EFC"
        )
        for column in ["temperature_C", "charge_c_rate", "discharge_c_rate"]:
            values = subset[column].dropna().value_counts().sort_index()
            formatted = ", ".join(
                f"{value:g}:{count}" for value, count in values.items()
            )
            print(f"  {column}: {formatted or 'no finite values'}")


def main():
    models = load_mlps()
    datasets = load_experimental_data()

    result_frames = []
    all_failures = []

    for source_name, dataset in datasets.items():
        chemistry = dataset["chemistry"]
        curves = dataset["curves"]
        metadata = dataset["metadata"]
        print(
            f"\nRunning inference for source {source_name} / {chemistry} "
            f"({len(metadata)} cells)"
        )
        chemistry_results, failures = run_chemistry_inference(
            chemistry,
            curves,
            metadata,
            models[chemistry],
        )
        # Ensure source provenance exists even for older VTC5A/VTC6 metadata.
        if "source_family" not in chemistry_results.columns:
            chemistry_results["source_family"] = source_name
        else:
            chemistry_results["source_family"] = (
                chemistry_results["source_family"].fillna(source_name)
            )
        result_frames.append(chemistry_results)
        all_failures.extend(
            (source_name, chemistry, cell_id, reason)
            for cell_id, reason in failures
        )

    results = pd.concat(result_frames, ignore_index=True)
    if results.empty:
        raise RuntimeError("No cells produced MLP features")

    # Calculate the three dependent stoichiometry features locally.
    results = add_derived_features(results)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(results)} rows to {OUTPUT}")

    print("\nProcessed cells by chemistry and stage:")
    print(results.groupby(["chemistry", "curve_type"])["cell_id"].nunique())
    if "source_family" in results.columns:
        print("\nProcessed cells by source and stage:")
        print(results.groupby(["source_family", "curve_type"])["cell_id"].nunique())

    for chemistry in CHEMISTRY_CONFIG:
        subset = results.loc[results["chemistry"] == chemistry]
        early_ids = set(subset.loc[subset["curve_type"] == "efc0", "cell_id"])
        late_ids = set(subset.loc[subset["curve_type"] == "efc50", "cell_id"])
        print(
            f"{chemistry}: unique={len(early_ids | late_ids)}, "
            f"paired={len(early_ids & late_ids)}, "
            f"missing_efc0={len(late_ids - early_ids)}, "
            f"missing_efc50={len(early_ids - late_ids)}"
        )

    if all_failures:
        print(f"\nInference failures: {len(all_failures)}")
        for source_name, chemistry, cell_id, reason in all_failures:
            print(f"  {source_name} / {chemistry} {cell_id}: {reason}")
    else:
        print("\nInference failures: 0")

    print_target_condition_summary(results)
    print_normalized_means(results)


if __name__ == "__main__":
    main()
