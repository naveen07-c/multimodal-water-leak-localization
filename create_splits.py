import pandas as pd
import numpy as np
from pathlib import Path

meta_df = pd.read_csv('/home/naveen/Pictures/agy_water_leak/metadata/02_master_metadata.csv')
exp_df = meta_df[~meta_df.experiment_id.str.contains('BACKGROUND_NOISE')].drop_duplicates('experiment_id').copy()

print(f"Total unique experimental scenarios: {len(exp_df)}")
print("\nExperiment distribution across conditions:")
print(pd.crosstab([exp_df.topology, exp_df.leak_type], [exp_df.flow_condition_code, exp_df.noise_condition]))

# Stratified Group Split
# Group by (leak_type, topology) to ensure balanced representation in Train, Val, Test
np.random.seed(42)

train_exps = []
val_exps = []
test_exps = []

# Group by (topology, leak_type)
for (top, leak), group in exp_df.groupby(['topology_code', 'leak_type_code']):
    exps = group['experiment_id'].tolist()
    np.random.shuffle(exps)
    n = len(exps)
    # n is either 6 (for BR/LO with ND_N, ND_NN, Trans_N, Trans_NN, 0.18_N, 0.47_N)
    # Target: ~60% train (3-4), ~20% val (1), ~20% test (1-2)
    n_val = max(1, int(round(n * 0.2)))
    n_test = max(1, int(round(n * 0.2)))
    n_train = n - n_val - n_test
    
    train_exps.extend(exps[:n_train])
    val_exps.extend(exps[n_train:n_train + n_val])
    test_exps.extend(exps[n_train + n_val:])

print(f"\nSplit Scenario Counts -> Train: {len(train_exps)}, Val: {len(val_exps)}, Test: {len(test_exps)} (Total: {len(train_exps)+len(val_exps)+len(test_exps)})")

# Map to full recording IDs and files
train_records = meta_df[meta_df.experiment_id.isin(train_exps)].copy()
val_records = meta_df[meta_df.experiment_id.isin(val_exps)].copy()
test_records = meta_df[meta_df.experiment_id.isin(test_exps)].copy()

print(f"File counts -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

# Verify NO overlap
overlap_tv = set(train_records.recording_id).intersection(set(val_records.recording_id))
overlap_tt = set(train_records.recording_id).intersection(set(test_records.recording_id))
overlap_vt = set(val_records.recording_id).intersection(set(test_records.recording_id))
assert len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0, "DATA LEAKAGE DETECTED!"

# Save split files in metadata/ and root
meta_dir = Path('/home/naveen/Pictures/agy_water_leak/metadata')
root_dir = Path('/home/naveen/Pictures/agy_water_leak')

train_records.to_csv(meta_dir / '03_train_ids.csv', index=False)
val_records.to_csv(meta_dir / '04_val_ids.csv', index=False)
test_records.to_csv(meta_dir / '05_test_ids.csv', index=False)

train_records.to_csv(root_dir / '03_train_ids.csv', index=False)
val_records.to_csv(root_dir / '04_val_ids.csv', index=False)
test_records.to_csv(root_dir / '05_test_ids.csv', index=False)

print("\nSaved 03_train_ids.csv, 04_val_ids.csv, 05_test_ids.csv successfully.")

# Print class distribution per split
print("\n--- Leak Type Distribution Across Splits (Scenario Level) ---")
df_split_summary = pd.DataFrame({
    'Train_Scenarios': exp_df[exp_df.experiment_id.isin(train_exps)]['leak_type_code'].value_counts(),
    'Val_Scenarios': exp_df[exp_df.experiment_id.isin(val_exps)]['leak_type_code'].value_counts(),
    'Test_Scenarios': exp_df[exp_df.experiment_id.isin(test_exps)]['leak_type_code'].value_counts(),
})
print(df_split_summary)

print("\n--- Leak Status Distribution (Binary: 0=No-Leak, 1=Leak) ---")
print("Train:", train_records.leak_status.value_counts(normalize=True).to_dict())
print("Val  :", val_records.leak_status.value_counts(normalize=True).to_dict())
print("Test :", test_records.leak_status.value_counts(normalize=True).to_dict())
