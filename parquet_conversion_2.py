import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import struct
from tqdm import tqdm


# -----------------------------
# Pass 1: Compute global baseline (mean)
# -----------------------------
def compute_global_baseline(
    emd_file_path,
    adc_divisor=1,
    chunk_size=500_000,
):
    total_sum = 0.0
    total_count = 0
    buffer = []

    with open(emd_file_path, "rb") as f:
        while True:
            data_bytes = f.read(8)
            if not data_bytes:
                break

            word = struct.unpack("<Q", data_bytes)[0]
            adc = (word & 0xFFFFFF) / adc_divisor
            buffer.append(adc)

            if len(buffer) >= chunk_size:
                arr = np.array(buffer)
                total_sum += arr.sum()
                total_count += len(arr)
                buffer.clear()

        if buffer:
            arr = np.array(buffer)
            total_sum += arr.sum()
            total_count += len(arr)

    return total_sum / total_count


# -----------------------------
# Pass 2: Stream + fill gaps + write parquet
# -----------------------------
def process_binary_to_parquet_streaming(
    emd_file_path,
    adc_divisor=1,
    dt=None,
    chunk_size=500_000,
    verbose=True,
):
    emd_file_path = Path(emd_file_path)
    parquet_file = emd_file_path.with_suffix(".parquet")

    if verbose:
        print("Pass 1: Computing global baseline...")

    baseline = compute_global_baseline(
        emd_file_path,
        adc_divisor=adc_divisor,
        chunk_size=chunk_size,
    )

    if verbose:
        print(f"Global baseline = {baseline}")
        print("Pass 2: Writing Parquet with gap filling...")

    # Pre-initialize ParquetWriter
    schema = pa.schema([
        ("timestamp", pa.float64()),
        ("adc_value", pa.float64()),
        ("is_filled", pa.bool_())
    ])
    writer = pq.ParquetWriter(parquet_file, schema)

    buffer = []
    prev_timestamp = None

    # estimate total records for tqdm
    total_records = emd_file_path.stat().st_size // 8

    with open(emd_file_path, "rb") as f, tqdm(total=total_records, disable=not verbose, desc="Processing Records") as pbar:
        while True:
            data_bytes = f.read(8)
            if not data_bytes:
                break

            word = struct.unpack("<Q", data_bytes)[0]
            timestamp = ((word >> 24) & 0x3FFFFFFFFF) * 0.1
            adc = (word & 0xFFFFFF) / adc_divisor

            # Gap detection
            if dt is not None and prev_timestamp is not None:
                delta = timestamp - prev_timestamp
                if delta > dt * 1.5:
                    n_missing = int(round(delta / dt)) - 1
                    if n_missing > 0:
                        for i in range(1, n_missing + 1):
                            missing_ts = prev_timestamp + i * dt
                            buffer.append((missing_ts, baseline, True))

            # Add real sample
            buffer.append((timestamp, adc, False))
            prev_timestamp = timestamp
            pbar.update(1)

            # Write chunk
            if len(buffer) >= chunk_size:
                df_chunk = pd.DataFrame(buffer, columns=["timestamp", "adc_value", "is_filled"])
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                writer.write_table(table)
                buffer.clear()

        # Write remaining buffer
        if buffer:
            df_chunk = pd.DataFrame(buffer, columns=["timestamp", "adc_value", "is_filled"])
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            writer.write_table(table)
            buffer.clear()

    writer.close()

    if verbose:
        print(f"Saved Parquet: {parquet_file}")

    return parquet_file


# -----------------------------
# Script runner
# -----------------------------
if __name__ == "__main__":
    import json
    from esamouil_functions import parse_txt_file

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

    dt = info["tia_summation_points"] * info["tia_sampling_ns"] * 1e-3
    adc_divisor = info.get("tia_summation_points", 1)

    process_binary_to_parquet_streaming(
        emd_file,
        adc_divisor=adc_divisor,
        dt=dt,
        chunk_size=500_000,
        verbose=True,
    )
