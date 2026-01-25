import pandas as pd
import os

RAW_PATH = "datasets/raw/measures_v2.csv"
OUTPUT_PATH = "datasets/processed/paderborn_profile_6.csv"
TARGET_PROFILE = 6  # The "Random Walk" profile for PINN training


def process_data():
    print(f"Loading dataset from {RAW_PATH}")

    # Check if file exists
    if not os.path.exists(RAW_PATH):
        print("Error: File not found!")
        return

    df = pd.read_csv(RAW_PATH)
    print(f"Original Size: {len(df)} rows")

    print(f"Filtering for Profile ID {TARGET_PROFILE}...")
    df_tiny = df[df['profile_id'] == TARGET_PROFILE].copy()

    # CLEANUP: Drop columns we don't need for the PINN
    # We only need: motor_speed, ambient, stator_yoke (temp), and torque (optional)
    keep_cols = ['profile_id', 'motor_speed',
                 'ambient', 'stator_yoke', 'torque']
    df_tiny = df_tiny[keep_cols]

    print(f"New Size: {len(df_tiny)} rows")

    df_tiny = df_tiny.reset_index(drop=True)

    # Create the Time Column
    # Formula: Time = Row_Index * (1 / Sampling_Freq)
    sampling_rate = 2.0  # Hz
    df_tiny['time'] = df_tiny.index * (1 / sampling_rate)

    # Reorder columns to have 'time' first (my OCD)
    cols = ['profile_id']+['time'] + \
        [col for col in df_tiny.columns if col not in ['time', 'profile_id']]
    df_tiny = df_tiny[cols]

    # SAVE
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_tiny.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_data()
