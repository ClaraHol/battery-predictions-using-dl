"""Prepare grouped battery features for the paper's zero-shot predictor.

Equivalent full cycles (EFC): supports 0 and 50.

The public entry point preserves the original six-value return signature.
The paper's legacy temperature-bin overlaps are retained intentionally.
"""

from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path('/work3/claho/battery_datasets/raw_feature_data')
DATA_FILES = {
    'test': 'test_data_efc{efc}_37groups.npy',
    'tongji': 'data_25r_efc{efc}.npy',
    'snl_a': 'data_m1a_efc{efc}.npy',
    'snl_b': 'data_hg2_efc{efc}.npy',
    'elt': 'data_mj1_efc{efc}.npy',
    'tum': 'data_vtc5a_efc{efc}.npy',
    'cmu': 'data_vtc6_efc{efc}.npy',
}
TRAIN_ORDER = ('tongji', 'snl', 'elt', 'tum', 'cmu')
CLUSTER_ORDER = ('tum', 'cmu', 'snl', 'elt', 'tongji')
TEMPERATURE_BINS = (5, 15, 20, 25, 30, 35, 40, 45, 50)
# Preserve original duplication: 35 C also contributes to the 30 C bin,
# and 45 C also contributes to the 40 C bin (training data only).
TRAIN_BIN_MEMBERS = {30: (30, 35), 40: (40, 45)}
DROP_COLUMNS = np.r_[28:56, 57, 59:74:2]
FEATURE_COUNT = 74
CURVE_WIDTH = 28
TEMPERATURE_COLUMN = 56
CHARGE_COLUMN_AFTER_REDUCTION = 28
DISCHARGE_COLUMN_AFTER_REDUCTION = 35


def _load_data(data_dir: Path):
    """Load both EFC measurements and EFC0 lifetime labels, per dataset."""
    loaded = {}
    for name, pattern in DATA_FILES.items():
        initial = np.load(data_dir / pattern.format(efc=0))
        later = np.load(data_dir / pattern.format(efc=50))
        loaded[name] = {
            0: initial[:, :FEATURE_COUNT].copy(),
            50: later.copy(),
            'labels': initial[:, FEATURE_COUNT:].copy(),
        }
    return loaded


def _round_to_step(values, step):
    return step * np.round(values / step)


def _prepare_features(features, labels, *, training, tum=False):
    """Return unclustered features/labels and the reduced clustering inputs.

    Unclustered features retain the full 74 columns as in the original script.
    """
    features = features.copy()
    labels = labels.copy()
    features[:, TEMPERATURE_COLUMN] = _round_to_step(
        features[:, TEMPERATURE_COLUMN] + 1e-9, 5
    )

    if tum:
        # TUM takes the mean nonzero C-rate across SOC intervals.
        for columns, destination in (((58, 60, 62, 64), 58),
                                     ((66, 68, 70, 72), 72)):
            rates = features[:, columns]
            valid = rates != 0
            counts = valid.sum(axis=1)
            sums = np.where(valid, rates, 0).sum(axis=1)
            means = np.divide(sums, counts, out=np.full(len(rates), np.nan),
                              where=counts != 0)
            rounded = _round_to_step(means + 1e-9, 0.5)
            rounded[rounded == 0] = 0.5
            features[:, destination] = rounded

    flagged_labels = np.column_stack((labels, np.ones(len(labels), dtype=int)))
    if training:
        # The original np.where(labels < 70)[0] keeps the first axis of matches.
        # Lifetimes are expected to be an (N, 1) array.
        keep = ~(labels < 70).any(axis=1)
        features = features[keep]
        flagged_labels = flagged_labels[keep]

    reduced = np.delete(features, DROP_COLUMNS, axis=1)
    temperature = reduced[:, 28].copy()
    reduced = np.delete(reduced, 28, axis=1)

    temperature[temperature == -5] = 5
    temperature[temperature == 13.75] = 15
    temperature[temperature == 32.5] = 35

    return features, flagged_labels, reduced, temperature


def _group_by_operating_conditions(reduced, labels, temperatures, *, training):
    """Group first 28 features + first 28 flagged-label columns by T and C-rates."""
    for t in TEMPERATURE_BINS:
        accepted = TRAIN_BIN_MEMBERS.get(t, (t,)) if training else (t,)
        selected = np.isin(temperatures, accepted)
        if not selected.any():
            continue
        subset = reduced[selected]
        subset_labels = labels[selected]
        charge = subset[:, CHARGE_COLUMN_AFTER_REDUCTION]
        discharge = subset[:, DISCHARGE_COLUMN_AFTER_REDUCTION]

        # TUM does not use this filter; the caller controls it.
        yield t, subset, subset_labels, charge, discharge


def _cluster(reduced, labels, temperatures, *, training, filter_rates):
    coordinates, groups = [], []
    for t, x, y, charge, discharge in _group_by_operating_conditions(
        reduced, labels, temperatures, training=training
    ):
        if filter_rates:
            valid = ((np.abs(charge) >= 0.001)
                     & (np.abs(discharge) >= 0.001)
                     & (np.abs(discharge) <= 10))
            x, y = x[valid], y[valid]
            charge, discharge = charge[valid], discharge[valid]
        if not len(x):
            continue

        rounded_pairs = np.column_stack((_round_to_step(charge, 0.5),
                                         _round_to_step(discharge, 0.5)))
        unique_pairs, inverse = np.unique(rounded_pairs, axis=0, return_inverse=True)
        observations = np.concatenate((x[:, :CURVE_WIDTH],
                                       y[:, :CURVE_WIDTH]), axis=1)
        for index, pair in enumerate(unique_pairs):
            coordinates.append(np.r_[t, pair])
            groups.append(observations[inverse == index])

    # Original sort is stable and by temperature only.
    order = sorted(range(len(coordinates)), key=lambda i: coordinates[i][0])
    return [coordinates[i] for i in order], [groups[i] for i in order]


def generate_data_list_group_mape(efc, data_dir=DEFAULT_DATA_DIR):
    """Create the paper's temperature/charge/discharge clusters.

    Returns (train_coordinates, train_groups, test_coordinates, test_groups,
             train_features, train_labels), matching the original API.

    train_features: full 74-column filtered/rounded features.
    train_labels: EFC0 labels with an appended flag column of ones.
    Each group contains the first 28 features + first 28 flagged-label columns.
    """
    if efc not in (0, 50):
        raise ValueError('efc must be 0 or 50')

    datasets = _load_data(Path(data_dir))
    datasets['snl'] = {
        efc: np.vstack((datasets['snl_a'][efc], datasets['snl_b'][efc])),
        'labels': np.vstack((datasets['snl_a']['labels'],
                             datasets['snl_b']['labels'])),
    }

    train_coordinates, train_groups = [], []
    train_features, train_labels = {}, {}
    clusters = {}
    for name in TRAIN_ORDER:
        features, labels, reduced, temperature = _prepare_features(
            datasets[name][efc], datasets[name]['labels'],
            training=True, tum=(name == 'tum')
        )
        train_features[name] = features
        train_labels[name] = labels
        coors, groups = _cluster(reduced, labels, temperature,
                                training=True, filter_rates=(name != 'tum'))
        clusters[name] = (coors, groups)

    for name in CLUSTER_ORDER:
        coors, groups = clusters[name]
        train_coordinates.extend(coors)
        train_groups.extend(groups)

    test = datasets['test']
    _, labels, reduced, temperature = _prepare_features(
        test[efc], test['labels'], training=False
    )
    test_coordinates, test_groups = _cluster(
        reduced, labels, temperature, training=False, filter_rates=False
    )

    # Sorting across datasets, preserving original dataset order within ties.
    order = sorted(range(len(train_coordinates)),
                   key=lambda i: train_coordinates[i][0])
    train_coordinates = [train_coordinates[i] for i in order]
    train_groups = [train_groups[i] for i in order]

    return (train_coordinates, train_groups, test_coordinates, test_groups,
            np.vstack([train_features[name] for name in TRAIN_ORDER]),
            np.vstack([train_labels[name] for name in TRAIN_ORDER]))
