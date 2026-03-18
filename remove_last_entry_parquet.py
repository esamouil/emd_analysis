import pandas as pd
from pathlib import Path

# ---- INPUT ----
parquet_file = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts/Z220126_IBM1_0008_part107.parquet")  # replace with your file
# ----------------

# Read parquet
df = pd.read_parquet(parquet_file)

if len(df) < 2:
    print("Not enough rows to trim.")
else:
    # Remove last row
    df = df.iloc[:-1]
    # Overwrite the original file
    df.to_parquet(parquet_file, index=False)
    print(f"Last row removed. New length: {len(df)}")