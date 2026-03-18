import pandas as pd
import struct
from tqdm import tqdm
import numpy as np
import scipy.fft as sp
import matplotlib.pyplot as plt
from pathlib import Path
import re
import fastparquet
import pyarrow 
import os
import plotly.express as px


def process_binary_to_csv(temp_file_path, count_to=None, verbose=True):
    """
    Reads an .emd file and saves the extracted timestamp and adc_value as a CSV.
    If the CSV already exists, loads it instead of regenerating.
    Returns the DataFrame for further use.
    """
    
    temp_file_path = Path(temp_file_path)
    filename_csv = temp_file_path.with_suffix(".csv")  # replace .emd with .csv

    if filename_csv.exists():
        if verbose:
            print(f"CSV already exists. Loading {filename_csv}...")
        df = pd.read_csv(filename_csv)
        return df

    if verbose:
        print(f"Processing {temp_file_path}...")

    data = []

    with open(temp_file_path, "rb") as file:
        counter = 0
        t0 = None
        total_records = int(count_to) if count_to is not None else None

        with tqdm(total=total_records, disable=not verbose, desc="Processing Records") as pbar:
            while True:
                data_bytes = file.read(8)
                if not data_bytes or (count_to is not None and counter >= count_to):
                    break
                counter += 1
                data_word = struct.unpack("<Q", data_bytes)[0]
                timestamp = (data_word >> 24) & 0x3FFFFFFFFF
                adc_value = data_word & 0xFFFFFF
                if t0 is None:
                    t0 = timestamp
                adjusted_timestamp = timestamp * 0.1  # in microseconds
                data.append({'timestamp': adjusted_timestamp, 'adc_value': adc_value})
                pbar.update(1)

    # after the loop
    if data:
        data = data[:-1]  # remove the last entry
    df = pd.DataFrame(data)
    df.to_csv(filename_csv, index=False)

    if verbose:
        print(f"Saved CSV: {filename_csv}")

    return df

def process_binary_to_csv_chunked(temp_file_path, count_to=None, verbose=True, chunk_size=100000):
    """
    Reads a large .emd file in chunks and saves timestamp and adc_value as CSV incrementally.
    """
    temp_file_path = Path(temp_file_path)
    filename_csv = temp_file_path.with_suffix(".csv")  # replace .emd with .csv

    if filename_csv.exists():
        if verbose:
            print(f"CSV already exists. Loading {filename_csv}...")
        return pd.read_csv(filename_csv)

    if verbose:
        print(f"Processing {temp_file_path} in chunks of {chunk_size}...")

    t0 = None
    counter = 0
    header_written = False

    with open(temp_file_path, "rb") as file, tqdm(total=count_to, disable=not verbose, desc="Processing Records") as pbar:
        while True:
            data = []
            for _ in range(chunk_size):
                data_bytes = file.read(8)
                if not data_bytes or (count_to is not None and counter >= count_to):
                    break
                counter += 1
                data_word = struct.unpack("<Q", data_bytes)[0]
                timestamp = (data_word >> 24) & 0x3FFFFFFFFF
                adc_value = data_word & 0xFFFFFF
                if t0 is None:
                    t0 = timestamp
                adjusted_timestamp = timestamp * 0.1  # in microseconds
                data.append({'timestamp': adjusted_timestamp, 'adc_value': adc_value})
            if not data:
                break
            # remove last row of the chunk (optional, like your original)
            data = data[:-1] if len(data) > 1 else data
            df_chunk = pd.DataFrame(data)
            df_chunk.to_csv(filename_csv, mode='a', index=False, header=not header_written)
            header_written = True
            pbar.update(len(data))

    if verbose:
        print(f"Saved CSV: {filename_csv}")

    return pd.read_csv(filename_csv)


def process_binary_to_parquet(temp_file_path, count_to=None, verbose=True):
    """
    Reads an .emd file and saves the extracted timestamp and adc_value as a Parquet file.
    If the Parquet file already exists, loads it instead of regenerating.
    Returns the DataFrame for further use.
    """
    
    temp_file_path = Path(temp_file_path)
    filename_parquet = temp_file_path.with_suffix(".parquet")  # replace .emd with .parquet

    if filename_parquet.exists():
        if verbose:
            print(f"Parquet already exists. Loading {filename_parquet}...")
        df = pd.read_parquet(filename_parquet)
        return df

    if verbose:
        print(f"Processing {temp_file_path}...")

    data = []

    with open(temp_file_path, "rb") as file:
        counter = 0
        t0 = None
        total_records = int(count_to) if count_to is not None else None

        with tqdm(total=total_records, disable=not verbose, desc="Processing Records") as pbar:
            while True:
                data_bytes = file.read(8)
                if not data_bytes or (count_to is not None and counter >= count_to):
                    break
                counter += 1
                data_word = struct.unpack("<Q", data_bytes)[0]
                timestamp = (data_word >> 24) & 0x3FFFFFFFFF
                adc_value = data_word & 0xFFFFFF
                if t0 is None:
                    t0 = timestamp
                adjusted_timestamp = timestamp * 0.1  # in microseconds
                data.append({'timestamp': adjusted_timestamp, 'adc_value': adc_value})
                pbar.update(1)

    # after the loop
    if data:
        data = data[:-1]  # remove the last entry
    df = pd.DataFrame(data)
    df.to_parquet(filename_parquet, index=False)

    if verbose:
        print(f"Saved Parquet: {filename_parquet}")

    return df



import pyarrow as pa
import pyarrow.parquet as pq

def process_binary_to_parquet_3(
    temp_file_path,
    count_to=None,
    verbose=True,
    chunk_size=500000,
    adc_divisor=1,
):
    """
    Reads an .emd file and saves the extracted timestamp and adc_value as a Parquet file.
    If the Parquet file already exists, loads it instead of regenerating.
    Drops the last record (matches original behavior).
    Optionally divides adc_value by adc_divisor before saving.
    Returns the DataFrame.
    """

    temp_file_path = Path(temp_file_path)
    filename_parquet = temp_file_path.with_suffix(".parquet")

    if filename_parquet.exists():
        if verbose:
            print(f"Parquet already exists. Loading {filename_parquet}...")
        return pd.read_parquet(filename_parquet)

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
                adc_value = (data_word & 0xFFFFFF) / adc_divisor  # <-- divide here

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

    return pd.read_parquet(filename_parquet)



def csv_to_parquet(csv_path, overwrite=False):
    csv_path = Path(csv_path)
    parquet_path = csv_path.with_suffix(".parquet")

    if parquet_path.exists() and not overwrite:
        print(f"Parquet already exists: {parquet_path}")
        return pd.read_parquet(parquet_path)

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Writing Parquet: {parquet_path}")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    return pd.read_parquet(parquet_path)




import pandas as pd

def read_ascii_emd(file_path):
    """
    Read an ASCII EMD file and extract timestamp and adc_value.
    
    Args:
        file_path (str): Path to the ASCII .emd file.
        
    Returns:
        pd.DataFrame: Columns ['timestamp', 'adc_value'], timestamps in microseconds.
    """
    timestamps = []
    adc_values = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                # column 2 is timestamp, column 4 is adc_value
                ts = float(parts[1]) * 0.1  # convert to microseconds
                val = float(parts[3])
                timestamps.append(ts)
                adc_values.append(val)

    return pd.DataFrame({
        "timestamp": timestamps,
        "adc_value": adc_values
    })

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



def baseline_shift(df, ranges):
    """
    Shift dataframe so it's centered at zero, using multiple percentage ranges
    to compute the baseline. Ignores gap-filled rows (is_filled==True) if column exists.
    """
    n = len(df)
    values = []

    for (p1, p2) in ranges:
        i1 = int(p1 * n)
        i2 = int(p2 * n)
        slice_df = df.iloc[i1:i2]

        # if 'is_filled' exists, ignore filled rows
        if "is_filled" in slice_df.columns:
            slice_df = slice_df.loc[~slice_df["is_filled"]]

        values.append(slice_df["adc_value"])

    all_vals = pd.concat(values)

    # compute baseline
    baseline = all_vals.mean()
    std_val  = all_vals.std()

    # shift original data
    df["adc_value"] = df["adc_value"] - baseline

    return baseline, std_val




def fft_dataframe(df, scale=1.0):
    """
    Compute FFT from a DataFrame containing 'timestamp' (µs) and 'adc_value'.
    Returns a new DataFrame with columns: frequency, magnitude, and also Nyquist frequency and deltaF.
    
    Parameters:
        df : DataFrame
            Must contain 'timestamp' and 'adc_value'.
        scale : float, optional
            Multiply ADC values by this factor before FFT (default=1.0).
    """
    signal = df['adc_value'].values * scale
    n = len(signal)

    if n < 2:
        print("Not enough samples.")
        return None

    # timestamp is in microseconds -> convert to seconds
    delta_t = np.diff(df['timestamp']).mean() * 1e-6  
    sampling_rate = 1.0 / delta_t

    # Nyquist frequency
    nyquist_freq = sampling_rate / 2

    # Frequency resolution
    deltaF = 1.0 / (n * delta_t)

    # FFT
    fft_vals = sp.fft(signal)

    # One-sided amplitude
    magnitudes = 2.0 * np.abs(fft_vals[:n//2]) / n

    # Frequency axis
    freqs = np.linspace(0, nyquist_freq, n//2)

    # Return as DataFrame along with Nyquist frequency and deltaF
    return pd.DataFrame({
        "frequency": freqs,
        "magnitude": magnitudes
    }), nyquist_freq, deltaF



def find_top_peaks_old(df_fft, n=10):
    """
    Returns the top n peaks from a frequency-magnitude dataframe.
    But it can be used to find the peaks at any dataframe.
    """
    if 'frequency' not in df_fft or 'magnitude' not in df_fft:
        print("Missing required columns.")
        return None

    # sort by amplitude, largest first
    peaks = df_fft.sort_values('magnitude', ascending=False).head(n)

    return peaks



def detect_fft_peaks(df_fft, neighborhood=1000, top_n=10):
    """
    Detect peaks in an FFT dataframe using neighborhood local-maxima method
    and calculate their FWHM.

    Parameters
    ----------
    df_fft : pd.DataFrame
        Must have columns ['frequency', 'magnitude'].
    neighborhood : int
        Number of points before and after each point to compare.
    top_n : int
        Number of largest peaks to keep.

    Returns
    -------
    df_peaks : pd.DataFrame
        Columns: ['frequency', 'magnitude', 'FWHM'] for top peaks.
    """

    freqs = df_fft["frequency"].to_numpy()
    mags = df_fft["magnitude"].to_numpy()

    # -- local maxima detection --
    peaks = []
    for i in range(neighborhood, len(mags) - neighborhood):
        if mags[i] > np.max(mags[i - neighborhood:i]) and mags[i] > np.max(mags[i + 1:i + 1 + neighborhood]):
            peaks.append(i)
    peaks = np.array(peaks)

    # -- keep only top N peaks --
    if len(peaks) > top_n:
        top_indices = np.argsort(mags[peaks])[::-1][:top_n]
        peaks = peaks[top_indices]

    # -- calculate FWHM --
    fwhm_list = []
    for idx in peaks:
        m_peak = mags[idx]
        half = m_peak / 2
        # search left
        left_idx = idx
        while left_idx > 0 and mags[left_idx] > half:
            left_idx -= 1
        # search right
        right_idx = idx
        while right_idx < len(mags) - 1 and mags[right_idx] > half:
            right_idx += 1
        fwhm = freqs[right_idx] - freqs[left_idx]
        fwhm_list.append(fwhm)

    # -- build dataframe --
    df_peaks = pd.DataFrame({
        "frequency": freqs[peaks],
        "magnitude": mags[peaks],
        "FWHM": fwhm_list
    }).sort_values("magnitude", ascending=False).reset_index(drop=True)

    return df_peaks





def plot_with_peaks(df_fft, peaks=None):
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(df_fft["frequency"], df_fft["magnitude"], label="FFT")

    if peaks is not None:
        ax.scatter(peaks["frequency"], peaks["magnitude"], color="red", zorder=5, label="Peaks")
        # optionally annotate
        for _, row in peaks.iterrows():
            ax.text(row["frequency"], row["magnitude"]*1.05, f'{row["frequency"]:.1f}', 
                    rotation=45, ha='center', fontsize=8)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT with Peaks")
    ax.legend()
    plt.tight_layout()

    return fig, ax





def plot_df_range(df, x_col, y_col, x_min=None, x_max=None, title=None, xlabel=None, ylabel=None):
    """
    Plots y_col vs x_col from df, limited to a specified x-axis range.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): column name for x-axis.
        y_col (str): column name for y-axis.
        x_min (float, optional): minimum x value to display.
        x_max (float, optional): maximum x value to display.
        title (str, optional): plot title.
        xlabel (str, optional): x-axis label.
        ylabel (str, optional): y-axis label.
    """
    mask = pd.Series(True, index=df.index)
    if x_min is not None:
        mask &= df[x_col] >= x_min
    if x_max is not None:
        mask &= df[x_col] <= x_max

    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(df.loc[mask, x_col], df.loc[mask, y_col])

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    return fig, ax

def plot_df_ranges(df, x_col, y_col, x_min=None, x_max=None, y_min=None, y_max=None, title=None, xlabel=None, ylabel=None):
    """
    Plots y_col vs x_col from df, limited to specified x and y axis ranges.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): column name for x-axis.
        y_col (str): column name for y-axis.
        x_min (float, optional): minimum x value to display.
        x_max (float, optional): maximum x value to display.
        y_min (float, optional): minimum y value to display.
        y_max (float, optional): maximum y value to display.
        title (str, optional): plot title.
        xlabel (str, optional): x-axis label.
        ylabel (str, optional): y-axis label.
    """
    mask = pd.Series(True, index=df.index)
    if x_min is not None:
        mask &= df[x_col] >= x_min
    if x_max is not None:
        mask &= df[x_col] <= x_max
    if y_min is not None:
        mask &= df[y_col] >= y_min
    if y_max is not None:
        mask &= df[y_col] <= y_max

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.loc[mask, x_col], df.loc[mask, y_col])

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    return fig, ax



def rebin(df, factor):
    """
    Rebin a histogram DataFrame by combining 'factor' consecutive rows.
    Keeps the original column names.

    Args:
        df (pd.DataFrame): DataFrame with two columns (x and y values).
        factor (int): Number of rows/bins to combine.

    Returns:
        pd.DataFrame: Rebinned histogram with same column names.
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame must have exactly 2 columns")

    col_x, col_y = df.columns
    x = df[col_x].values
    y = df[col_y].values
    n = len(y)

    # truncate to multiple of factor
    n_new = (n // factor) * factor
    x = x[:n_new]
    y = y[:n_new]

    # rebin
    y_rebinned = y.reshape(-1, factor).sum(axis=1)
    x_rebinned = x.reshape(-1, factor).mean(axis=1)

    return pd.DataFrame({col_x: x_rebinned, col_y: y_rebinned})



def rebin_by_width(df, new_width, xcol=None, ycol=None):
    # auto-detect if not provided
    if xcol is None or ycol is None:
        if df.shape[1] != 2:
            raise ValueError("Specify xcol and ycol if DataFrame has != 2 columns")
        xcol, ycol = df.columns

    x = df[xcol].values
    y = df[ycol].values

    xmin = x.min()
    xmax = x.max()

    edges = np.arange(xmin, xmax + new_width, new_width)

    new_x = []
    new_y = []

    for i in range(len(edges) - 1):
        low = edges[i]
        high = edges[i+1]

        mask = (x >= low) & (x < high)
        if not mask.any():
            continue

        new_x.append((low + high) / 2)
        new_y.append(y[mask].sum())

    return pd.DataFrame({xcol: new_x, ycol: new_y})

def rebin_by_width_centered(df, new_width, xcol=None, ycol=None):
    if xcol is None or ycol is None:
        if df.shape[1] != 2:
            raise ValueError("Specify xcol and ycol if DataFrame has != 2 columns")
        xcol, ycol = df.columns

    x = df[xcol].values
    y = df[ycol].values

    xmin = x.min()
    xmax = x.max()

    # Compute number of bins
    n_bins = int(np.ceil((xmax - xmin) / new_width))
    
    # Compute bin edges: centered bins
    edges = xmin + np.arange(n_bins + 1) * new_width
    centers = edges[:-1] + new_width / 2

    # Assign each x to a bin index
    bin_idx = np.searchsorted(edges, x, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # clip edges

    # Sum values per bin
    sums = np.bincount(bin_idx, weights=y, minlength=n_bins)

    # Remove empty bins if you want
    mask = sums > 0
    return pd.DataFrame({xcol: centers[mask], ycol: sums[mask]})

def rebin_by_width_centered_power_preserved(df, new_width, xcol=None, ycol=None):
    if xcol is None or ycol is None:
        if df.shape[1] != 2:
            raise ValueError("Specify xcol and ycol if DataFrame has != 2 columns")
        xcol, ycol = df.columns

    x = df[xcol].values
    y = df[ycol].values

    xmin = x.min()
    xmax = x.max()

    # Compute number of bins
    n_bins = int(np.ceil((xmax - xmin) / new_width))
    
    # Compute bin edges: centered bins
    edges = xmin + np.arange(n_bins + 1) * new_width
    centers = edges[:-1] + new_width / 2

    # Assign each x to a bin index
    bin_idx = np.searchsorted(edges, x, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Sum squares per bin (power)
    sums = np.bincount(bin_idx, weights=y**2, minlength=n_bins)

    # Take square root to get amplitude-like value
    amplitudes = np.sqrt(sums)

    # Remove empty bins
    mask = amplitudes > 0
    return pd.DataFrame({xcol: centers[mask], ycol: amplitudes[mask]})

def rebin_by_width_centered_power_preserved_2(df, new_width, xcol=None, ycol=None):
    if xcol is None or ycol is None:
        if df.shape[1] != 2:
            raise ValueError("Specify xcol and ycol if DataFrame has != 2 columns")
        xcol, ycol = df.columns

    x = df[xcol].values
    y = df[ycol].values

    xmin = x.min()
    xmax = x.max()

    # Anchor bins to integer multiples of new_width
    start_edge = np.floor(xmin / new_width) * new_width - new_width / 2
    end_edge   = np.ceil(xmax / new_width) * new_width + new_width / 2

    edges = np.arange(start_edge, end_edge + new_width, new_width)
    centers = edges[:-1] + new_width / 2
    n_bins = len(centers)

    # Assign x values to bins
    bin_idx = np.searchsorted(edges, x, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Sum power (square) per bin
    sums = np.bincount(bin_idx, weights=y**2, minlength=n_bins)

    # Convert back to amplitude-like quantity
    amplitudes = np.sqrt(sums)

    # Remove empty bins
    mask = amplitudes > 0

    return pd.DataFrame({
        xcol: centers[mask],
        ycol: amplitudes[mask]
    })

def _hms_to_seconds(hms):
    parts = hms.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = map(int, parts)
        return h*3600 + m*60 + s
    except ValueError:
        return None


def parse_txt_file(path):
    out = {
        "detector_name": None,
        "event_binary": None,
        "nominal_time_s": None,
        "real_time_s": None,
        "absolute_time": None,
        "hv0": None,
        "chopper_freq_hz": None,
        "tia_sampling_ns": None,
        "tia_summation_points": None,
    }

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("Detector Name"):
                _, val = line.split(":", 1)
                out["detector_name"] = val.strip()

            elif line.startswith("Event Data file binary"):
                _, val = line.split(":", 1)
                try:
                    out["event_binary"] = int(val.strip())
                except ValueError:
                    out["event_binary"] = None

            elif line.startswith("Nominal Time"):
                _, val = line.split(":", 1)
                val = val.strip()
                out["nominal_time_s"] = _hms_to_seconds(val)

            elif line.startswith("Real Time"):
                _, val = line.split(":", 1)
                val = val.strip()
                out["real_time_s"] = _hms_to_seconds(val)

            elif line.startswith("Absolute Time"):
                _, val = line.split(":", 1)
                try:
                    out["absolute_time"] = int(val.strip())
                except ValueError:
                    out["absolute_time"] = None

            elif line.startswith("HV 0"):
                # e.g. "HV 0:  891 V"
                m = re.search(r"HV\s*0\s*:\s*([0-9]+)", line)
                if m:
                    out["hv0"] = int(m.group(1))

            elif line.startswith("Chopper Frequency"):
                # e.g. "Chopper Frequency [Hz]: 0.000"
                _, val = line.split(":", 1)
                try:
                    out["chopper_freq_hz"] = float(val.strip())
                except ValueError:
                    # fallback: find a float in the line
                    m = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", line)
                    out["chopper_freq_hz"] = float(m.group(1)) if m else None

            elif line.startswith("Tia sampling period"):
                # prefer the "=> 6256 ns" style
                m = re.search(r"=>\s*([0-9]+)\s*ns", line)
                if m:
                    out["tia_sampling_ns"] = int(m.group(1))
                else:
                    # fallback: any number followed by 'ns'
                    m2 = re.search(r"([0-9]+)\s*ns", line)
                    if m2:
                        out["tia_sampling_ns"] = int(m2.group(1))
                    else:
                        # last resort: grab the second integer on the line if present
                        nums = re.findall(r"(\d+)", line)
                        if len(nums) >= 2:
                            out["tia_sampling_ns"] = int(nums[1])

            elif line.startswith("Tia summation points"):
                # e.g. "Tia summation points: 4  => in total ..."
                m = re.search(r"Tia summation points\s*:\s*([0-9]+)", line)
                if m:
                    out["tia_summation_points"] = int(m.group(1))

    return out


def calculate_noise_metrics_from_single_df(df):
    """
    Calculate noise metrics from a single FFT DataFrame.

    Args:
        df (pd.DataFrame): Must have columns 'frequency' and 'magnitude'.

    Returns:
        pd.DataFrame: Single-row DataFrame with calculated metrics.
    """
    xf = df['frequency'].values
    yf = df['magnitude'].values

    total_power = np.sum(yf**2)
    low_power = np.sum(yf[xf < 1000]**2)
    mid_power = np.sum(yf[(xf >= 1000) & (xf <= 10000)]**2)
    high_power = np.sum(yf[xf > 10000]**2)
    peak_frequency = xf[np.argmax(yf)]

    return pd.DataFrame([{
        'total_power': total_power,
        'low_power': low_power,
        'mid_power': mid_power,
        'high_power': high_power,
        'peak_frequency': peak_frequency
    }])


def filter_df(df, ta, tb):
    """
    Slice a DataFrame to timestamps in [ta, tb] seconds, then shift so the first timestamp is 0.
    
    Args:
        df (pd.DataFrame): Must have columns ['timestamp', 'adc_value'].
        ta (float): Start time in seconds.
        tb (float): End time in seconds.
        
    Returns:
        pd.DataFrame: Filtered and shifted DataFrame.
    """
    # Convert seconds to microseconds
    ta_us = ta * 1e6
    tb_us = tb * 1e6
    
    # Filter
    filtered = df[(df['timestamp'] >= ta_us) & (df['timestamp'] <= tb_us)].copy()
    
    # Shift timestamps so first one is 0
    if not filtered.empty:
        filtered['timestamp'] = filtered['timestamp'] - filtered['timestamp'].iloc[0]
    
    return filtered


def save_plot(fig, name, output_dir, save_outputs=True):
    """
    Save a matplotlib figure if saving is enabled.
    
    Args:
        fig: matplotlib.figure.Figure object
        name: filename without path
        output_dir: path to save the plot
        save_outputs: bool, whether to save or not
    """
    if save_outputs:
        fig_path = os.path.join(output_dir, name)
        fig.savefig(fig_path, bbox_inches="tight")





# chopper specific

def split_measurements_by_threshold(df, adc_threshold, id_col="meas_id"):
    """
    Add a measurement ID column to the DataFrame and drop separator rows.

    Args:
        df (pd.DataFrame): must contain 'adc_value'
        adc_threshold (float): values above this mark a new measurement
        id_col (str): name of the measurement ID column

    Returns:
        pd.DataFrame: the same DataFrame, modified in place
    """
    if "adc_value" not in df.columns:
        raise ValueError("DataFrame must contain 'adc_value'")

    # identify separators
    is_sep = df["adc_value"] > adc_threshold

    # assign measurement ids
    df[id_col] = is_sep.cumsum()

    # drop separator rows
    df.drop(index=df[is_sep].index, inplace=True)

    # clean index
    df.reset_index(drop=True, inplace=True)

    return df

def split_measurements_by_time(df, measurement_us, time_col="timestamp", id_col="meas_id"):
    """
    Split a DataFrame into measurements of fixed duration in milliseconds.

    Args:
        df (pd.DataFrame): must contain the time_col
        measurement_ms (float): measurement width in milliseconds
        time_col (str): column containing the timestamp (assumed in µs)
        id_col (str): name of the measurement ID column to create

    Returns:
        pd.DataFrame: the same DataFrame with a new measurement ID column
    """
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column")

    # Convert measurement width to microseconds

    # Assign measurement IDs based on integer division
    df[id_col] = (df[time_col] // measurement_us).astype(int)

    # Optional: reset index
    df.reset_index(drop=True, inplace=True)

    return df





def shift_measurements_timestamps(df):
    """
    Shift timestamps in each measurement so the first timestamp is 0.
    """
    # get first timestamp per meas_id
    t0 = df.groupby("meas_id")["timestamp"].transform("first")
    
    # subtract in one vectorized operation
    df["timestamp"] -= t0
    
    return df

def get_peak_timestamps(df):
    """
    For each measurement in df (excluding meas_id 0), get the timestamp of the highest adc_value.

    Args:
        df (pd.DataFrame): must have 'timestamp', 'adc_value', 'meas_id'

    Returns:
        pd.DataFrame: single-column DataFrame with peak timestamps
    """
    peak_times = []

    for mid, dfi in df.groupby("meas_id"):
        if mid == 0:
            continue  # skip the first measurement

        idx_max = dfi["adc_value"].idxmax()
        peak_times.append(dfi.loc[idx_max, "timestamp"])

    return pd.DataFrame({"peak_timestamp": peak_times})

def get_two_peaks_per_measurement(df):
    """
    For each measurement (excluding meas_id 0), split it in half and get the timestamp
    of the maximum adc_value in each half.

    Args:
        df (pd.DataFrame): must have 'timestamp', 'adc_value', 'meas_id'

    Returns:
        pd.DataFrame: single-column DataFrame with peak timestamps
    """
    peak_times = []

    for mid, dfi in df.groupby("meas_id"):
        if mid == 0:
            continue  # skip first measurement

        n = len(dfi)
        half = n // 2

        # first half
        first_half = dfi.iloc[:half]
        idx_max1 = first_half["adc_value"].idxmax()
        peak_times.append(dfi.loc[idx_max1, "timestamp"])

        # second half
        second_half = dfi.iloc[half:]
        idx_max2 = second_half["adc_value"].idxmax()
        peak_times.append(dfi.loc[idx_max2, "timestamp"])

    return pd.DataFrame({"peak_timestamp": peak_times})



def interactive_plot(df, downsample=None, x_range=None, extra_scatter=None):
    """
    Generic interactive Plotly line plot.
    First column → x
    Second column → y
    Only shows plot if save_outputs is False (controlled outside the function).

    Parameters:
    - df: pandas DataFrame (first column = x, second column = y)
    - downsample: int, take every nth row for plotting (optional)
    - x_range: tuple (min, max) to limit x-axis (optional)
    - extra_scatter: dict with keys 'x', 'y', 'name' to overlay points (optional)
    """
    # Downsample if requested
    df_plot = df.iloc[::downsample, :] if downsample else df

    x_col = df_plot.columns[0]
    y_col = df_plot.columns[1]

    fig = px.line(df_plot, x=x_col, y=y_col)

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
    )

    if x_range:
        fig.update_xaxes(range=x_range)

    if extra_scatter:
        fig.add_scatter(
            x=extra_scatter["x"],
            y=extra_scatter["y"],
            mode="markers",
            name=extra_scatter.get("name", "Points")
        )

    fig.show()


def detect_sparks_old(df, std_val, coarse_step=100, threshold_sigma=8, time_window=10000):
    """
    Detect sparks in a baseline-shifted ADC dataframe and return start, peak, and duration.

    Returns a list of dicts with:
        'start_time', 'start_adc', 'peak_time', 'peak_adc', 'duration'
    """
    threshold_val = threshold_sigma * std_val

    # --- Coarse scan ---
    df_coarse = df.iloc[::coarse_step]
    coarse_flags = df_coarse[abs(df_coarse["adc_value"]) > threshold_val]
    if coarse_flags.empty:
        return []

    # --- Cluster consecutive candidates ---
    interesting_times = df.loc[coarse_flags.index, "timestamp"].sort_values().to_list()
    clusters = []
    current_cluster = [interesting_times[0]]
    for t in interesting_times[1:]:
        if t - current_cluster[-1] <= time_window:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)

    # --- Analyze each cluster ---
    sparks = []

    for cluster in clusters:
        print("\nDetected cluster (coarse timestamps):")
        print(cluster)

        first_idx = df.index.get_loc(df[df["timestamp"] == cluster[0]].index[0])
        last_idx  = df.index.get_loc(df[df["timestamp"] == cluster[-1]].index[0])

        # Backtrack for true spark start
        start_slice = max(0, first_idx - coarse_step)
        back_slice = df.iloc[start_slice:first_idx + 1]
        above_thresh = back_slice[abs(back_slice["adc_value"]) > threshold_val]
        if not above_thresh.empty:
            true_start_idx = above_thresh.index[0]
        else:
            true_start_idx = df.index[first_idx]

        start_time = df.loc[true_start_idx, "timestamp"]
        start_adc  = df.loc[true_start_idx, "adc_value"]

        # Find spark peak
        peak_start = max(0, first_idx - (coarse_step - 1))
        peak_slice = df.iloc[peak_start:last_idx + 1]
        peak_idx   = peak_slice["adc_value"].abs().idxmax()
        peak_time  = df.loc[peak_idx, "timestamp"]
        peak_adc   = df.loc[peak_idx, "adc_value"]

        # Forward-track for true spark end
        end_slice = df.iloc[last_idx : min(last_idx + coarse_step + 1, len(df))]
        above_thresh_end = end_slice[abs(end_slice["adc_value"]) > threshold_val]
        if not above_thresh_end.empty:
            true_end_idx = above_thresh_end.index[-1]
        else:
            true_end_idx = df.index[last_idx]

        end_time = df.loc[true_end_idx, "timestamp"]
        duration = end_time - start_time

        sparks.append({
            "start_time": start_time,
            "start_adc": start_adc,
            "peak_time": peak_time,
            "peak_adc": peak_adc,
            "duration": duration
        })

    return sparks




def detect_sparks(df, std_val, trigger_sigma=6, boundary_sigma=4,
                  time_window=10000, min_points=3, t_stable=20000):
    """
    Detect sparks in ADC data.
    
    Parameters:
    - df : pandas.DataFrame with columns 'timestamp' and 'adc_value'
    - std_val : float, standard deviation of the baseline-subtracted signal
    - trigger_sigma : float, threshold in sigma to detect spark points
    - boundary_sigma : float, threshold in sigma to detect stable regions
    - time_window : int, maximum separation in µs to group points into a cluster
    - min_points : int, minimum points in a cluster to be considered a spark
    - t_stable : int, time in µs to define stable region before/after spark
    
    Returns:
    - sparks : dict, each key is 'spark_1', 'spark_2', ... with a dict containing:
        - 'cluster_indices' : indices of points in df forming the spark
        - 'n_points' : number of points
        - 'peak_time' : timestamp of peak
        - 'peak_adc' : ADC value at peak
        - 't_start' : start time of spark (after stable region detection)
        - 't_end' : end time of spark (after stable region detection)
        - 'duration' : duration of spark in µs
    """
    ts = df["timestamp"].to_numpy()
    adc = df["adc_value"].to_numpy()
    
    threshold_trigger = trigger_sigma * std_val
    threshold_boundary = boundary_sigma * std_val
    
    over_trigger = np.abs(adc) > threshold_trigger
    over_boundary = np.abs(adc) > threshold_boundary
    
    flagged_indices = np.where(over_trigger)[0]
    if len(flagged_indices) == 0:
        return {}
    
    # Build clusters
    clusters = []
    current_cluster = [flagged_indices[0]]
    for idx in flagged_indices[1:]:
        dt = ts[idx] - ts[current_cluster[-1]]
        if dt <= time_window:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)
    
    clusters = [c for c in clusters if len(c) >= min_points]
    
    # Precompute stable regions
    stable_before = np.zeros_like(adc, dtype=bool)
    stable_after  = np.zeros_like(adc, dtype=bool)
    
    for i in range(len(ts)):
        start_idx = np.searchsorted(ts, ts[i]-t_stable)
        stable_before[i] = np.all(~over_boundary[start_idx:i+1])
        end_idx   = np.searchsorted(ts, ts[i]+t_stable)
        stable_after[i] = np.all(~over_boundary[i:end_idx])
    
    # Extract spark info
    sparks = {}
    
    for i, cluster in enumerate(clusters):
        cluster_ts = ts[cluster]
        cluster_adc = adc[cluster]
        
        # Peak
        peak_idx = cluster[np.argmax(np.abs(cluster_adc))]
        peak_time = ts[peak_idx]
        peak_adc_val = adc[peak_idx]
        
        # t_start
        idx_before = cluster[0]
        candidates = np.where(stable_before[:idx_before])[0]
        t_start = ts[candidates[-1]] if len(candidates) > 0 else ts[0]
        
        # t_end
        idx_after = cluster[-1]
        candidates = np.where(stable_after[idx_after:])[0]
        t_end = ts[idx_after + candidates[0]] if len(candidates) > 0 else ts[-1]
        
        duration = max(0, t_end - t_start)
        
        sparks[f"spark_{i+1}"] = {
            "cluster_indices": cluster,
            "n_points": len(cluster),
            "peak_time": peak_time,
            "peak_adc": peak_adc_val,
            "t_start": t_start,
            "t_end": t_end,
            "duration": duration
        }
    
    return sparks

def detect_sparks_in_range(df, std_val, trigger_sigma=6, boundary_sigma=4,
                  time_window=10000, min_points=3, t_stable=20000,
                  start_idx=None, end_idx=None):
    """
    Detect sparks in ADC data within a specific index range of the dataframe.

    Parameters:
    - df : pandas.DataFrame with columns 'timestamp' and 'adc_value'
    - std_val : float, standard deviation of the baseline-subtracted signal
    - trigger_sigma : float, threshold in sigma to detect spark points
    - boundary_sigma : float, threshold in sigma to detect stable regions
    - time_window : int, maximum separation in µs to group points into a cluster
    - min_points : int, minimum points in a cluster to be considered a spark
    - t_stable : int, time in µs to define stable region before/after spark
    - start_idx : int, optional, start index of the dataframe to analyze
    - end_idx : int, optional, end index of the dataframe to analyze

    Returns:
    - sparks : dict, each key is 'spark_1', 'spark_2', ... with details
    """
    # default to full range
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df) - 1

    # use views of the NumPy arrays (no new DataFrame)
    ts = df["timestamp"].to_numpy()[start_idx:end_idx+1]
    adc = df["adc_value"].to_numpy()[start_idx:end_idx+1]

    threshold_trigger = trigger_sigma * std_val
    threshold_boundary = boundary_sigma * std_val

    over_trigger = np.abs(adc) > threshold_trigger
    over_boundary = np.abs(adc) > threshold_boundary

    flagged_indices = np.where(over_trigger)[0]
    if len(flagged_indices) == 0:
        return {}

    # Build clusters
    clusters = []
    current_cluster = [flagged_indices[0]]
    for idx in flagged_indices[1:]:
        dt = ts[idx] - ts[current_cluster[-1]]
        if dt <= time_window:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)

    clusters = [c for c in clusters if len(c) >= min_points]

    # Precompute stable regions
    stable_before = np.zeros_like(adc, dtype=bool)
    stable_after  = np.zeros_like(adc, dtype=bool)

    for i in range(len(ts)):
        s_idx = np.searchsorted(ts, ts[i]-t_stable)
        stable_before[i] = np.all(~over_boundary[s_idx:i+1])
        e_idx = np.searchsorted(ts, ts[i]+t_stable)
        stable_after[i] = np.all(~over_boundary[i:e_idx])

    # Extract spark info
    sparks = {}
    for i, cluster in enumerate(clusters):
        cluster_ts = ts[cluster]
        cluster_adc = adc[cluster]

        # Peak
        peak_idx = cluster[np.argmax(np.abs(cluster_adc))]
        peak_time = ts[peak_idx]
        peak_adc_val = adc[peak_idx]

        # t_start
        idx_before = cluster[0]
        candidates = np.where(stable_before[:idx_before])[0]
        t_start = ts[candidates[-1]] if len(candidates) > 0 else ts[0]

        # t_end
        idx_after = cluster[-1]
        candidates = np.where(stable_after[idx_after:])[0]
        t_end = ts[idx_after + candidates[0]] if len(candidates) > 0 else ts[-1]

        duration = max(0, t_end - t_start)

        # adjust cluster_indices to original df indices
        sparks[f"spark_{i+1}"] = {
            "cluster_indices": [start_idx + idx for idx in cluster],
            "n_points": len(cluster),
            "peak_time": peak_time,
            "peak_adc": peak_adc_val,
            "t_start": t_start,
            "t_end": t_end,
            "duration": duration
        }

    return sparks

def detect_sparks_in_range_fast(df, std_val, trigger_sigma=6, boundary_sigma=4,
                                time_window=10000, min_points=3, t_stable=20000,
                                start_idx=None, end_idx=None, max_duration=100_000):
    """
    Detect sparks in a specific range with speed optimizations and maximum duration filtering.

    Parameters:
    ----------
    df : pandas.DataFrame
        Must have 'timestamp' (µs) and 'adc_value' columns.
    std_val : float
        Standard deviation of the baseline-subtracted signal.
    trigger_sigma : float, default=6
        Threshold in sigma to detect spark points.
    boundary_sigma : float, default=4
        Threshold in sigma to detect stable regions around sparks.
    time_window : int, default=10000
        Max separation (µs) between points in a cluster.
    min_points : int, default=3
        Minimum points in a cluster to count as a spark.
    t_stable : int, default=20000
        Duration (µs) used to define stable regions before/after a spark.
    start_idx : int, optional
        Start index in df to analyze.
    end_idx : int, optional
        End index in df to analyze.
    max_duration : int, default=100_000
        Maximum duration (µs) for a spark; longer clusters are ignored.

    Returns:
    -------
    sparks : dict
        Each key is 'spark_1', 'spark_2', ... containing cluster indices, peak info, start/end times, and duration.
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df) - 1

    ts = df["timestamp"].to_numpy()[start_idx:end_idx+1]
    adc = df["adc_value"].to_numpy()[start_idx:end_idx+1]

    threshold_trigger = trigger_sigma * std_val
    threshold_boundary = boundary_sigma * std_val

    over_trigger = np.abs(adc) > threshold_trigger
    over_boundary = np.abs(adc) > threshold_boundary

    flagged_indices = np.where(over_trigger)[0]
    if len(flagged_indices) == 0:
        return {}

    # ---- cluster building ----
    clusters = []
    current_cluster = [flagged_indices[0]]
    for idx in flagged_indices[1:]:
        if ts[idx] - ts[current_cluster[-1]] <= time_window:
            current_cluster.append(idx)
        else:
            if len(current_cluster) >= min_points:
                clusters.append(current_cluster)
            current_cluster = [idx]

    if len(current_cluster) >= min_points:
        clusters.append(current_cluster)

    if not clusters:
        return {}

    # ---- FAST stable detection using cumulative sum ----
    boundary_int = over_boundary.astype(int)
    cumsum = np.cumsum(boundary_int)

    def no_boundary(i1, i2):
        if i1 > i2:
            return True
        if i1 == 0:
            return cumsum[i2] == 0
        return (cumsum[i2] - cumsum[i1-1]) == 0

    sparks = {}

    for i, cluster in enumerate(clusters):
        cluster_ts = ts[cluster]
        cluster_adc = adc[cluster]

        peak_idx = cluster[np.argmax(np.abs(cluster_adc))]
        peak_time = ts[peak_idx]
        peak_adc_val = adc[peak_idx]

        # ---- find t_start ----
        idx_before = cluster[0]
        t_start = ts[0]
        for j in range(idx_before, -1, -1):
            s_idx = np.searchsorted(ts, ts[j] - t_stable)
            if no_boundary(s_idx, j):
                t_start = ts[j]
                break

        # ---- find t_end ----
        idx_after = cluster[-1]
        t_end = ts[-1]
        for j in range(idx_after, len(ts)):
            e_idx = np.searchsorted(ts, ts[j] + t_stable)
            if no_boundary(j, e_idx-1):
                t_end = ts[j]
                break

        duration = max(0, t_end - t_start)

        # ---- skip clusters exceeding max_duration ----
        if duration > max_duration:
            continue

        sparks[f"spark_{i+1}"] = {
            "cluster_indices": [start_idx + idx for idx in cluster],
            "n_points": len(cluster),
            "peak_time": peak_time,
            "peak_adc": peak_adc_val,
            "t_start": t_start,
            "t_end": t_end,
            "duration": duration
        }

    return sparks



def detect_constant_noise_ranges(df, window_size=100000, rel_tol=0.2, persist=3, merge_tol=0.1):
    """
    Detect constant-noise ranges ignoring short spikes, merge consecutive low-std ranges,
    and assign transition windows to the side with larger std.

    Args:
        df (pd.DataFrame): baseline-shifted
        window_size (int): points per window
        rel_tol (float): relative tolerance for std
        persist (int): number of consecutive windows the std must exceed rel_tol to break a range
        merge_tol (float): max std value for consecutive ranges to be merged

    Returns:
        List of tuples: [(start_idx, end_idx, local_std), ...]
    """
    n = len(df)
    if n == 0:
        return []

    window_stds = []
    indices = []

    # compute std per window
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window_stds.append(df["adc_value"].iloc[start:end].std())
        indices.append((start, end-1))

    # initial range detection with persist
    ranges = []
    current_start, current_end = indices[0]
    current_std = window_stds[0]
    spike_count = 0
    pending_start, pending_std = None, None

    for (s_idx, e_idx), w_std in zip(indices[1:], window_stds[1:]):
        rel_diff = abs(w_std - current_std) / current_std if current_std != 0 else float('inf')

        if rel_diff <= rel_tol:
            # continue current range
            current_end = e_idx
            spike_count = 0
            pending_start, pending_std = None, None
        else:
            # start a pending range if not already
            if pending_start is None:
                pending_start, pending_std = s_idx, w_std
                spike_count = 1
            else:
                spike_count += 1

            # check if persisted
            if spike_count >= persist:
                # assign the transition window to the side with larger std
                if pending_std > current_std:
                    ranges.append((current_start, pending_start-1, current_std))
                    current_start, current_end = pending_start, e_idx
                    current_std = pending_std
                else:
                    ranges.append((current_start, s_idx-1, current_std))
                    current_start, current_end = s_idx, e_idx
                    current_std = w_std
                spike_count = 0
                pending_start, pending_std = None, None

        # update current_std as weighted average
        current_std = ((current_std * (current_end - current_start + 1)) + (w_std * (e_idx - s_idx + 1))) / ((current_end - current_start + 1) + (e_idx - s_idx + 1))

    ranges.append((current_start, current_end, current_std))

    # merge consecutive small-std ranges
    merged_ranges = []
    if not ranges:
        return merged_ranges

    current_start, current_end, current_std = ranges[0]
    for start, end, std_val in ranges[1:]:
        if current_std <= merge_tol and std_val <= merge_tol:
            # merge
            current_end = end
            # update std as weighted average
            current_std = ((current_std * (current_end - current_start + 1)) + (std_val * (end - start + 1))) / ((current_end - current_start + 1) + (end - start + 1))
        else:
            merged_ranges.append((current_start, current_end, current_std))
            current_start, current_end, current_std = start, end, std_val

    merged_ranges.append((current_start, current_end, current_std))
    return merged_ranges



