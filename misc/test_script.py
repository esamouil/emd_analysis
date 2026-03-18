#%% This script is to test functions and troubleshoot stuff
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import matplotlib.pyplot as plt
import os
from pathlib import Path
#%% === Plot Style Config ===
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%%
#%% 
# Load the config file
with open("config.json", "r") as f:
    config = json.load(f)

#%%
# Convert to Path object
folder = Path(config["data_folder_path"])
txt_files = [folder / config["single_file_name"]]
# find matching .emd file(s) by some convention, e.g. same stem
stem = txt_files[0].stem
emd_files = list(folder.glob(f"{stem}*.emd"))

print("EMD File:", [f.name for f in emd_files])
print("TXT File:", [f.name for f in txt_files])    

#%%
# Get parameters from the txt file.
info = parse_txt_file(txt_files[0])
print(info)

#%% Define the function to turn the binary file into a parquet file

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import struct

def convert_binary_to_parquet(
    temp_file_path,
    count_to=None,
    verbose=True,
    chunk_size=500_000,
):
    """
    Reads an .emd file and saves the extracted timestamp and adc_value as a Parquet file.
    Drops the last record (matches original behavior).
    Only handles the conversion, does NOT return a DataFrame.
    """

    temp_file_path = Path(temp_file_path)
    filename_parquet = temp_file_path.with_suffix(".parquet")

    if filename_parquet.exists():
        if verbose:
            print(f"Parquet already exists: {filename_parquet}")
        return

    if verbose:
        print(f"Processing {temp_file_path}...")

    writer = None
    buffer = []
    pending = None  # holds last record so we can drop it

    with open(temp_file_path, "rb") as file:
        counter = 0
        total_records = int(count_to) if count_to is not None else None

        with tqdm(total=total_records, disable=not verbose, desc="Processing Records") as pbar:
            while True:
                data_bytes = file.read(8)
                if not data_bytes or (count_to is not None and counter >= count_to):
                    break

                counter += 1
                data_word = struct.unpack("<Q", data_bytes)[0]

                timestamp = ((data_word >> 24) & 0x3FFFFFFFFF) * 0.1
                adc_value = data_word & 0xFFFFFF

                current = (timestamp, adc_value)

                if pending is not None:
                    buffer.append(pending)

                pending = current
                pbar.update(1)

                if len(buffer) >= chunk_size:
                    df = pd.DataFrame(buffer, columns=["timestamp", "adc_value"])
                    table = pa.Table.from_pandas(df, preserve_index=False)

                    if writer is None:
                        writer = pq.ParquetWriter(filename_parquet, table.schema)

                    writer.write_table(table)
                    buffer.clear()

    # write remaining buffered rows (but NOT `pending`)
    if buffer:
        df = pd.DataFrame(buffer, columns=["timestamp", "adc_value"])
        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(filename_parquet, table.schema)

        writer.write_table(table)

    if writer:
        writer.close()

    if verbose:
        print(f"Saved Parquet: {filename_parquet}")

#%% Function to read the parquet file into a dataframe

import pandas as pd
from pathlib import Path

def read_parquet_to_df(parquet_file_path, verbose=True):
    """
    Reads a Parquet file and returns it as a pandas DataFrame.
    """
    parquet_file_path = Path(parquet_file_path)

    if not parquet_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

    if verbose:
        print(f"Loading Parquet file: {parquet_file_path}...")

    df = pd.read_parquet(parquet_file_path)

    if verbose:
        print(f"Loaded {len(df)} rows from {parquet_file_path}")

    return df


#%% testing
