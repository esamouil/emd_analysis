import pandas as pd
import struct
from tqdm import tqdm
import numpy as np
import scipy.fft as sp
import matplotlib.pyplot as plt
from pathlib import Path
import re



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



def baseline_shift(df, ranges):
    """
    Shift dataframe so it's centered at zero, using multiple percentage ranges
    to compute the baseline.

    Args:
        df (pd.DataFrame): must contain 'adc_value'
        ranges (list of tuples): e.g. [(0.05, 0.25), (0.40, 0.50), (0.75, 0.95)]
                                 Each tuple = (start_percent, end_percent)

    Returns:
        baseline (float): computed mean, and also the standard deviation of the entries.
    """
    n = len(df)
    values = []

    for (p1, p2) in ranges:
        i1 = int(p1 * n)
        i2 = int(p2 * n)
        values.append(df["adc_value"].iloc[i1:i2])

    # combine all slices
    all_vals = pd.concat(values)

    # compute baseline
    baseline = all_vals.mean()
    std_val  = all_vals.std()

    # shift original data
    df["adc_value"] = df["adc_value"] - baseline

    return baseline, std_val




def fft_dataframe(df):
    """
    Compute FFT from a DataFrame containing 'timestamp' (µs) and 'adc_value'.
    Returns a new DataFrame with columns: frequency, magnitude.
    """
    signal = df['adc_value'].values
    n = len(signal)

    if n < 2:
        print("Not enough samples.")
        return None

    # timestamp is in microseconds -> convert to seconds
    delta_t = np.diff(df['timestamp']).mean() * 1e-6  
    sampling_rate = 1.0 / delta_t

    # FFT
    fft_vals = sp.fft(signal)

    # One-sided amplitude
    magnitudes = 2.0 * np.abs(fft_vals[:n//2]) / n

    # Frequency axis
    freqs = np.linspace(0, sampling_rate/2, n//2)

    # Return as DataFrame
    return pd.DataFrame({
        "frequency": freqs,
        "magnitude": magnitudes
    })



def find_top_peaks(df_fft, n=10):
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





def plot_with_peaks(df_fft, peaks=None):
    plt.figure(figsize=(10,5))
    plt.plot(df_fft["frequency"], df_fft["magnitude"], label="FFT")

    if peaks is not None:
        plt.scatter(peaks["frequency"], peaks["magnitude"], color="red", zorder=5, label="Peaks")
        # optionally annotate
        for _, row in peaks.iterrows():
            plt.text(row["frequency"], row["magnitude"]*1.05, f'{row["frequency"]:.1f}', 
                     rotation=45, ha='center', fontsize=8)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("FFT with Peaks")
    plt.legend()
    plt.tight_layout()
    plt.show()




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
    
    plt.figure(figsize=(10,5))
    plt.plot(df.loc[mask, x_col], df.loc[mask, y_col])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


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
    high_power = np.sum(yf[xf > 10000]**2)
    peak_frequency = xf[np.argmax(yf)]

    return pd.DataFrame([{
        'total_power': total_power,
        'low_power': low_power,
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

