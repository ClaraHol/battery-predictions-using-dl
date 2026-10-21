import numpy as np
from pathlib import Path
from cleaned_battery_features import generate_data_list_group_mape

DATA_ROOT = Path(f"/work3/claho/battery_datasets/raw_feature_data")

(
    train_coordinates,
    train_groups,
    test_coordinates,
    test_groups,
    train_features,
    train_labels,
) = generate_data_list_group_mape(efc=0)

print(f"groups: {train_groups[0].shape}, metadata: {train_coordinates[0].shape}, features: {train_features[0].shape}, label: {train_labels[0].shape}")
print(train_groups[0])
print(train_labels[0])