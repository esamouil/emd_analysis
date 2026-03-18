#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import os
from pathlib import Path
from scipy.signal import find_peaks
import plotly.express as px
import plotly.io as pio

#%% Configuring plot style
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')
#for interactive plots:
pio.renderers.default = "notebook"
#%% 
# Load the config file
with open("config.json", "r") as f:
    config = json.load(f)
save_outputs = config.get("save_outputs", False)

#%%
# Convert to Path object
folder = Path(config["data_folder_path"])
txt_files = [folder / config["single_file_name"]]
# find matching .emd file(s) by some convention, e.g. same stem
stem = txt_files[0].stem
emd_files = list(folder.glob(f"{stem}*.emd"))

#%% Saving outputs setup
output_dir = os.path.join(config["data_folder_path"], emd_files[0].stem)

if save_outputs:
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

# subdirectories
adc_dir = os.path.join(output_dir, "adc")
voltage_dir = os.path.join(output_dir, "voltage")
adc_sparks_dir = os.path.join(adc_dir, "sparks")
voltage_sparks_dir = os.path.join(voltage_dir, "sparks")
fft_dir = os.path.join(output_dir, "fft")

# create subdirectories
for d in [adc_dir, voltage_dir, adc_sparks_dir, voltage_sparks_dir, fft_dir]:
    os.makedirs(d, exist_ok=True)

if save_outputs:
    log_file_path = os.path.join(output_dir, "analysis.log")
    log_file = open(log_file_path, "w")
    import sys

    # Helper class to write to both terminal and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.__stdout__, log_file)

#%% Print the file names 
print("=============== Run ===============")
print("EMD File:", [f.name for f in emd_files])
print("TXT File:", [f.name for f in txt_files])
print("===================================\n")    

#%%
# Get parameters from the txt file.
info = parse_txt_file(txt_files[0])
print("=============== Run Info ===============")
for k, v in info.items():
    print(f"{k}: {v}")
print("========================================\n")

#%% File conversion and import to dataframe.
print("=============== Data Import ===============")
if info["event_binary"] :   # event_binary is 1 for binary
    df = process_binary_to_parquet_3(emd_files[0], count_to=config["entry_limit"], verbose=True, adc_divisor=info.get("tia_summation_points", 1))
else :                      # if its 0 then its txt
    df = read_ascii_emd(emd_files[0])

print(df.head())
print(df.tail())
print("========================================\n")

# %%
# plot the dataframe

fig, ax = plt.subplots(figsize=(16,4), dpi=100)

ax.plot(df["timestamp"], df["adc_value"])
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Raw file")

# Little info box in top-right corner
textstr = f'Detector: {info["detector_name"]}'
ax.text(
    0.35, 0.95, textstr, transform=ax.transAxes,
    fontsize=12, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
)

fig.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "raw_file", adc_dir, save_outputs)

#region Voltage plot
fig, ax = plt.subplots(figsize=(16,4), dpi=100)

# plot ADC values scaled on the fly
ax.plot(df["timestamp"], df["adc_value"] * config['adc_to_voltage'], lw=0.8)

ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("Voltage (μV)")
ax.set_title("Raw file")

# little info box
textstr = f'Detector: {info["detector_name"]}'
ax.text(
    0.35, 0.95, textstr, transform=ax.transAxes,
    fontsize=12, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
)

fig.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "raw_file_voltage", voltage_dir, save_outputs)


#endregion


#%%
# filter_df does two things, it first shirt the timestamps to start from 0 and it filters the values to be up to a limit
df = filter_df(df, 0, config['timestamp_lim'])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df.head())
print(df.tail())
print("=============================================\n")


ranges =[(0.0,1.0)]
baseline, std_val = baseline_shift(df, ranges)
print("=============== Baseline Calculation ===============")
print("Ranges (Percentage of range):")
for r in ranges:
    print(f"{r[0]*100} - {r[1]*100}")
print(f"Baseline: {baseline:.2f}")
print(f"Standard Deviation: {std_val:.2f}")
print(f"Baseline (μV): {(baseline*config['adc_to_voltage']):.2f}")
print(f"Standard Deviation (μV): {(std_val*config['adc_to_voltage']):.2f}")
print("===================================================\n")



#%%
# plot the shifted dataframe

fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"],lw=0.5)
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Shifted by baseline")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"shifted by baseline",adc_dir, save_outputs)
#region Voltage Plot
fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"]*config['adc_to_voltage'],lw=0.5)
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("Voltage (μV)")
ax.set_title("Shifted by baseline")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"shifted by baseline",voltage_dir, save_outputs)
# %%


# %%
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



# %%
# Detect constant-noise ranges with rolling std
noise_ranges = detect_constant_noise_ranges(df, window_size=100, rel_tol=0.1, persist=100)

print("Detected constant-noise ranges:")
for start, end, local_std in noise_ranges:
    print(f"Start: {start}, End: {end}, Points: {end-start+1}, Local std: {local_std:.3f}")

# Plot ADC signal with constant-noise ranges highlighted
fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"], lw=0.5, label="ADC Signal")

# Highlight constant-noise ranges with alternating colors
colors = ['yellow', 'darkgray']

for i, (start_idx, end_idx, local_std) in enumerate(noise_ranges):
    ax.axvspan(
        df["timestamp"].iloc[start_idx],
        df["timestamp"].iloc[end_idx],
        color=colors[i % 2],
        alpha=0.3
    )

ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("ADC Signal with Constant-Noise Ranges Highlighted")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "constant_noise_ranges", adc_dir, save_outputs)
# %%
# Detect sparks using local std per constant-noise range
all_sparks = {}

for i, (start_idx, end_idx, local_std) in enumerate(noise_ranges):
    if local_std < 0.01:  # skip flat regions
        continue

    sparks = detect_sparks_new(
        df,
        std_val=local_std,
        trigger_sigma=6,
        boundary_sigma=4,
        time_window=10000,
        min_points=3,
        t_stable=20000,
        start_idx=start_idx,
        end_idx=end_idx
    )

    # rename keys to include range index so they stay unique
    for k, v in sparks.items():
        all_sparks[f"range{i+1}_{k}"] = v

print(f"Detected {len(all_sparks)} sparks across {len(noise_ranges)} noise ranges")


# %%
trigger_sigma=6
boundary_sigma=4

n_padding = 0.4  # fraction of duration to pad before and after

if all_sparks:
    for i, (key, s) in enumerate(all_sparks.items(), start=1):
        spark_start = s['t_start']
        spark_end   = s['t_end']
        duration    = s['duration']

        start_time = spark_start - n_padding * duration
        end_time   = spark_end   + n_padding * duration

        df_zoom = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_zoom['timestamp'], df_zoom['adc_value'], lw=0.8)

        # Thresholds (ADC units)
        trigger_thr  = trigger_sigma * std_val
        boundary_thr = boundary_sigma * std_val

        ax.axhline(+trigger_thr,  color='blue',  linestyle='--')
        ax.axhline(-trigger_thr,  color='blue',  linestyle='--')
        ax.axhline(+boundary_thr, color='green', linestyle=':')
        ax.axhline(-boundary_thr, color='green', linestyle=':')

        # Add labels on the lines
        x_pos = df_zoom['timestamp'].min()  # left side of plot
        ax.text(x_pos, +trigger_thr,  f'+{trigger_sigma}σ', color='blue',  va='bottom', ha='left', fontsize=10)
        ax.text(x_pos, -trigger_thr,  f'-{trigger_sigma}σ', color='blue',  va='top',    ha='left', fontsize=10)
        ax.text(x_pos, +boundary_thr, f'+{boundary_sigma}σ', color='green', va='bottom', ha='left', fontsize=10)
        ax.text(x_pos, -boundary_thr, f'-{boundary_sigma}σ', color='green', va='top',    ha='left', fontsize=10)

        ax.axvspan(spark_start, spark_end, color='orange', alpha=0.3, label='Spark Duration')
        ax.scatter(s['peak_time'], s['peak_adc'], color='red', zorder=5, label='Spark Peak')

        mid_time = (spark_start + spark_end) / 2
        y_max = df_zoom['adc_value'].max()
        ax.text(mid_time, y_max, f"Duration: {duration:.1f} µs",
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlabel("Timestamp (µs)")
        ax.set_ylabel("ADC value")
        ax.set_title(f"Spark {i} Zoomed View")
        ax.legend()
        plt.tight_layout()

        plt.show() if not save_outputs else None
        save_plot(fig, f"spark_{i}", adc_sparks_dir, save_outputs)
# %%
