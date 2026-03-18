#%% Imports
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import struct
from tqdm import tqdm
import json
from esamouil_functions import *

#%% Vectorized gap filling
def fill_gaps_vectorized(df, dt, baseline_method="median"):
    """
    Fill large gaps in timestamp using expected dt (µs),
    inserting baseline ADC values.
    Adds column 'is_filled' only if there are gaps.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    ts = df["timestamp"].values
    delta = np.diff(ts)
    gap_idx = np.where(delta > dt * 1.5)[0]

    if len(gap_idx) == 0:
        # no gaps → return df as-is, no extra column
        return df

    # only now create the column
    df["is_filled"] = False

    baseline = (
        df["adc_value"].median()
        if baseline_method == "median"
        else df["adc_value"].mean()
    )

    chunks = []
    last = 0

    for idx in gap_idx:
        chunks.append(df.iloc[last : idx + 1])

        t0 = ts[idx]
        t1 = ts[idx + 1]
        n_missing = int(round((t1 - t0) / dt)) - 1

        if n_missing > 0:
            missing_ts = t0 + np.arange(1, n_missing + 1) * dt
            missing_adc = np.full(n_missing, baseline)

            missing_df = pd.DataFrame({
                "timestamp": missing_ts,
                "adc_value": missing_adc,
                "is_filled": True,
            })

            chunks.append(missing_df)

        last = idx + 1

    chunks.append(df.iloc[last:])
    return pd.concat(chunks, ignore_index=True)

#%% Binary → Parquet with ADC correction and gap filling
def process_binary_to_parquet_with_gaps(
    emd_file_path,
    adc_divisor=1,
    dt=None,
    chunk_size=500_000,
    verbose=True,
):
    emd_file_path = Path(emd_file_path)
    parquet_file = emd_file_path.with_suffix(".parquet")

    if parquet_file.exists():
        if verbose:
            print(f"Parquet already exists. Loading {parquet_file}...")
        return pd.read_parquet(parquet_file)

    if verbose:
        print(f"Processing {emd_file_path}...")

    writer = None
    buffer = []
    pending = None

    with open(emd_file_path, "rb") as f, tqdm(
        desc="Processing Records", disable=not verbose
    ) as pbar:
        while True:
            data_bytes = f.read(8)
            if not data_bytes:
                break

            data_word = struct.unpack("<Q", data_bytes)[0]
            timestamp = ((data_word >> 24) & 0x3FFFFFFFFF) * 0.1  # µs
            adc_value = (data_word & 0xFFFFFF) / adc_divisor

            current = (timestamp, adc_value)

            if pending is not None:
                buffer.append(pending)
            pending = current
            pbar.update(1)

            if len(buffer) >= chunk_size:
                df_chunk = pd.DataFrame(
                    buffer, columns=["timestamp", "adc_value"]
                )
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_file, table.schema)
                writer.write_table(table)
                buffer.clear()

    if buffer:
        df_chunk = pd.DataFrame(buffer, columns=["timestamp", "adc_value"])
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    if verbose:
        print("Loading full Parquet to fill gaps...")

    df = pd.read_parquet(parquet_file)

    # IMPORTANT: drop known-bad last record (binary format invariant)
    df = df.iloc[:-1].reset_index(drop=True)

    if dt is not None:
        df = fill_gaps_vectorized(df, dt=dt)

    df.to_parquet(parquet_file, index=False)

    if verbose:
        print(f"Saved Parquet with gaps filled: {parquet_file}")

    return df

#%% ---- Script execution ----
if __name__ == "__main__":
    with open("config_parquet_conversion.json") as f:
        config = json.load(f)

    folder = Path(config["data_folder_path"])
    txt_file = folder / config["single_file_name"]

    info = parse_txt_file(txt_file)

    stem = txt_file.stem
    emd_files = list(folder.glob(f"{stem}*.emd"))
    if not emd_files:
        raise FileNotFoundError("No matching .emd files found")

    emd_file = emd_files[0]

    dt = (
        info["tia_summation_points"]
        * info["tia_sampling_ns"]
        * 1e-3
    )  # µs

    adc_divisor = info.get("tia_summation_points", 1)

    df = process_binary_to_parquet_with_gaps(
        emd_file,
        adc_divisor=adc_divisor,
        dt=dt,
        chunk_size=500_000,
        verbose=True,
    )

    print(df.head())
    print(df.tail())

# %%
