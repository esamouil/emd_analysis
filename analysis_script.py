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
from scipy.optimize import curve_fit
import numpy as np

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
    dt = (
    info["tia_summation_points"]
    * info["tia_sampling_ns"]
    * 1e-3
    )  # µs
    df=fill_gaps_vectorized(df, dt=dt)

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

ax.plot(df["timestamp"], df["adc_value"] * config['adc_to_voltage'])
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("Voltage (μV)")
ax.set_title("Raw file")

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
# filter_df does two things, it first shifts the timestamps to start from 0 and it filters the values to be up to a limit
df = filter_df(df, 0, config['timestamp_lim'])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df.head())
print(df.tail())
print("=============================================\n")




#%%  Plotting the last chunk of 120000 lines, and if its less thatn that, the remainder
fraction = 0.02  # last 2%
total_len = len(df)

start_idx = int((1 - fraction) * total_len)
df_tail = df.iloc[start_idx:]  # view only, df unchanged

x = range(len(df_tail))  # local line number

fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(x, df_tail["adc_value"], lw=0.5)
ax.set_xlabel("Line number (last 2%)")
ax.set_ylabel("ADC value")
ax.set_title("Last 2% of file")
ax.set_xlim(0, 120_000)   # <-- fixed axis

plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "last_2percent", adc_dir, save_outputs)

#region Voltage plot
fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(x, df_tail["adc_value"] * config['adc_to_voltage'], lw=0.5)
ax.set_xlabel("Line number (last 2%)")
ax.set_ylabel("Voltage (μV)")
ax.set_title("Last 2% of file")
ax.set_xlim(0, 120_000)   # <-- fixed axis

plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "last_2percent_voltage", voltage_dir, save_outputs)
#endregion

# %%
# calculate the baseline using some range
# the range is declared in respect of the percentage of the full range. Can put any number of entries in ranges list

# ranges = [
#     (0.00, 0.25),
#     (0.25, 0.75),
#     (0.75, 1.0)
# ]
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
ax.set_title("Run shifted by baseline")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"shifted by baseline",adc_dir, save_outputs)
#region Voltage Plot
fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"]*config['adc_to_voltage'],lw=0.5)
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("Voltage (μV)")
ax.set_title("Run shifted by baseline")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"shifted by baseline",voltage_dir, save_outputs)
#endregion

#interactive_plot(df, downsample=100) if not save_outputs else None

#%% histogram of the values


# Gaussian function like ROOT's TF1("gaus")
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Use the same logic as baseline_shift for values
if "is_filled" in df.columns:
    values_to_fit = df.loc[~df["is_filled"], "adc_value"].to_numpy() * config["adc_to_voltage"]
else:
    values_to_fit = df["adc_value"].to_numpy() * config["adc_to_voltage"]

# Define bin width and edges centered on 0
bin_width = 31.25  # adjust to your preference
max_abs = np.ceil(np.abs(values_to_fit).max())
bins = np.arange(-max_abs - bin_width/2, max_abs + bin_width, bin_width)  # edges like -0.5,0.5,1.5,...

bin_left = bins[:-1]
bin_right = bins[1:]
bin_centers = (bin_left + bin_right)/2

# Histogram counts
counts, _ = np.histogram(values_to_fit, bins=bins)

# Fit Gaussian to histogram
p0 = [counts.max(), 0, np.std(values_to_fit)]  # initial guess: amplitude, mean, sigma
popt, _ = curve_fit(gauss, bin_centers, counts, p0=p0)
A, mu, sigma = popt

# Plot histogram + Gaussian fit
fig, ax = plt.subplots(figsize=(10,4))

# Histogram
ax.hist(values_to_fit, bins=bins, alpha=0.6, edgecolor='k', label='Data')

# Gaussian fit
ax.plot(bin_centers, gauss(bin_centers, A, mu, sigma), 'r-', lw=2, label='Gaussian fit')

# Add text box with stats
textstr = f"μ = {mu:.3e}\nσ = {sigma:.3e}\nA = {A:.1f}"
ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

ax.set_title("Signal Values Histogram")
ax.set_xlabel("Signal (μV)")
ax.set_ylabel("Counts")
ax.legend()
fig.tight_layout()

# Show and optionally save
fig.show() if not save_outputs else None
save_plot(fig, "histogram", voltage_dir, save_outputs)

# Chi² / NDF calculation
# Use Poisson errors for histogram counts: sqrt(counts)
errors = np.sqrt(counts)
errors[errors == 0] = 1.0  # avoid division by zero

# Compute chi²
chi2 = np.sum(((counts - gauss(bin_centers, *popt)) / errors)**2)
ndf = len(counts) - len(popt)  # n_bins - n_parameters
chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

print("\n===== Gaussian Fit Statistics =====")
print(f"Amplitude A = {A:.2f}")
print(f"Mean μ      = {mu:.3e}")
print(f"Std σ       = {sigma:.3e}")
print(f"Chi²        = {chi2:.2f}")
print(f"NDF         = {ndf}")
print(f"Chi² / NDF  = {chi2_ndf:.2f}")
print(f"dstd%       = {100*((sigma/config['adc_to_voltage'])-std_val)/std_val:.2f}")
print("===================================")

# Free memory
plt.close(fig)                 # closes the figure
del values_to_fit, counts, bins, bin_left, bin_right, bin_centers, popt, A, mu, sigma, errors, chi2, ndf, chi2_ndf, textstr
#%% check if baseline std is 'constant' to then proceed to spark detection
# --- Check if baseline noise is roughly constant ---

n_chunks = 10
chunk_size = len(df) // n_chunks

stds = []

for i in range(n_chunks):
    start = i * chunk_size
    end   = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
    stds.append(df["adc_value"].iloc[start:end].std())

stds = np.array(stds)

std_mean = stds.mean()
std_rel_spread = stds.std() / std_mean  # relative variation
print("================== std metrics =================")
print("STD per chunk:", np.round(stds, 2))
print(f"Relative STD variation: {std_rel_spread:.3f}")
if std_rel_spread < 0.05:   # 5% variation threshold
    print("Noise roughly stable")
else:
    print("Noise not stable")
print("================================================\n")

#%% Detect ranges of stable noise

noise_ranges = detect_constant_noise_ranges(df, window_size=100, rel_tol=0.1, persist=100)

print("================== Constant noise ranges ==================")
print(f"{'Start Ind.':>12} {'End Ind.':>12} {'Points':>12} {'Local std':>12}")
for start, end, local_std in noise_ranges:
    print(f"{start:12d} {end:12d} {end-start+1:12d} {local_std:12.2f}")
print("===========================================================\n")
#%% plot the ranges

# Plot Voltage signal with constant-noise ranges highlighted
fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"]*config['adc_to_voltage'], lw=0.5, label="ADC Signal")

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
ax.set_ylabel("Voltage (μV)")
ax.set_title("Signal with Constant-Noise Ranges Highlighted")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "constant_noise_ranges", voltage_dir, save_outputs)

#%% Detect abnormalities

trigger_sigma = 6
boundary_sigma = 4

sparks = {}
spark_counter = 1

# Loop over detected constant-noise ranges
for start_idx, end_idx, local_std in noise_ranges:

    # skip nearly flat regions if needed
    if local_std < 0.01:
        continue

    range_sparks = detect_sparks_in_range_fast(
        df,
        std_val=local_std,
        trigger_sigma=trigger_sigma,
        boundary_sigma=boundary_sigma,
        time_window=10000,
        min_points=4,
        t_stable=10000,
        start_idx=start_idx,
        end_idx=end_idx,
        max_duration=100_000
    )

    # keep global numbering spark_1, spark_2, ...
    for _, spark_data in range_sparks.items():
        sparks[f"spark_{spark_counter}"] = spark_data
        spark_counter += 1


n_sparks = len(sparks)
run_time_s = info['real_time_s']
sparks_per_s = n_sparks / run_time_s if run_time_s > 0 else 0

print("======================== Spark Info ========================")
print(f"Total Abnormalities Detected: {n_sparks}")
print(f"Abnormalities per Second: {sparks_per_s:.2f}")
print(f"{'Spark':<6} {'Points':>6} {'Duration (µs)':>15} {'Peak (ADC)':>12} {'Peak (µV)':>12} {'Peak Time (µs)':>15}")

for key, s in sparks.items():
    print(f"{key:<6} {s['n_points']:>6} {s['duration']:>15.2f} "
          f"{s['peak_adc']:>12.2f} "
          f"{(s['peak_adc']*config['adc_to_voltage']):>12.2f} "
          f"{s['peak_time']:>15.2f}")

print("============================================================\n")

#%%

#%% Plot each spark (non-interactive) and save
n_padding = 0.4  # fraction of duration to pad before and after

if sparks:
    for i, (key, s) in enumerate(sparks.items(), start=1):
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

        ax.axvspan(spark_start, spark_end, color='orange', alpha=0.3, label='Approx. Duration')
        ax.scatter(s['peak_time'], s['peak_adc'], color='red', zorder=5, label='Max Peak')

        mid_time = (spark_start + spark_end) / 2
        y_max = df_zoom['adc_value'].max()
        ax.text(mid_time, y_max, f"Duration: {duration:.1f} µs",
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlabel("Timestamp (µs)")
        ax.set_ylabel("ADC value")
        ax.set_title(f"Abnormal Event {i} Zoomed View")
        ax.legend()
        plt.tight_layout()

        plt.show() if not save_outputs else None
        save_plot(fig, f"spark_{i}", adc_sparks_dir, save_outputs)

#region Voltage Plots
if sparks:
    for i, (key, s) in enumerate(sparks.items(), start=1):
        spark_start = s['t_start']
        spark_end   = s['t_end']
        duration    = s['duration']

        start_time = spark_start - n_padding * duration
        end_time   = spark_end   + n_padding * duration

        df_zoom = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

        fig, ax = plt.subplots(figsize=(12,4))
        adc_to_v = config['adc_to_voltage']
        ax.plot(df_zoom['timestamp'], df_zoom['adc_value']*adc_to_v, lw=0.8)

        # Thresholds (voltage)
        trigger_thr_v  = trigger_sigma  * std_val * adc_to_v
        boundary_thr_v = boundary_sigma * std_val * adc_to_v

        ax.axhline(+trigger_thr_v,  color='blue',  linestyle='--')
        ax.axhline(-trigger_thr_v,  color='blue',  linestyle='--')
        ax.axhline(+boundary_thr_v, color='green', linestyle=':')
        ax.axhline(-boundary_thr_v, color='green', linestyle=':')

        # Add labels on the lines
        x_pos = df_zoom['timestamp'].min()
        ax.text(x_pos, +trigger_thr_v,  f'+{trigger_sigma}σ', color='blue',  va='bottom', ha='left', fontsize=10)
        ax.text(x_pos, -trigger_thr_v,  f'-{trigger_sigma}σ', color='blue',  va='top',    ha='left', fontsize=10)
        ax.text(x_pos, +boundary_thr_v, f'+{boundary_sigma}σ', color='green', va='bottom', ha='left', fontsize=10)
        ax.text(x_pos, -boundary_thr_v, f'-{boundary_sigma}σ', color='green', va='top',    ha='left', fontsize=10)

        ax.axvspan(spark_start, spark_end, color='orange', alpha=0.3, label='Approx. Duration')
        ax.scatter(s['peak_time'], s['peak_adc']*adc_to_v, color='red', zorder=5, label='Max Peak')

        mid_time = (spark_start + spark_end) / 2
        y_max = (df_zoom['adc_value']*adc_to_v).max()
        ax.text(mid_time, y_max, f"Duration: {duration:.1f} µs",
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlabel("Timestamp (µs)")
        ax.set_ylabel("Voltage (µV)")
        ax.set_title(f"Abnormal Event {i} Zoomed View")
        ax.legend()
        plt.tight_layout()

        plt.show() if not save_outputs else None
        save_plot(fig, f"spark_{i}_voltage", voltage_sparks_dir, save_outputs)
    #endregion

#%% collective spark plot
if sparks:
    #=================== ADC Plot ===================
    fig, ax = plt.subplots(figsize=(16,4), dpi=100)

    # Plot full ADC signal
    ax.plot(df['timestamp'], df['adc_value'], lw=0.5, label='ADC Signal')

    # Plot thresholds per detected constant-noise range
    for start_idx, end_idx, local_std in noise_ranges:
        t_start = df['timestamp'].iloc[start_idx]
        t_end = df['timestamp'].iloc[end_idx]

        trigger_thr = trigger_sigma * local_std
        boundary_thr = boundary_sigma * local_std

        # Thick, high-contrast lines
        ax.hlines([+trigger_thr, -trigger_thr], t_start, t_end, color='blue', linestyle='--', lw=3, alpha=1)
        ax.hlines([+boundary_thr, -boundary_thr], t_start, t_end, color='green', linestyle=':', lw=3, alpha=1)

    # Side labels outside the plot
    x_side = df['timestamp'].iloc[-1] + 0.01*(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])
    ax.text(x_side, +trigger_thr, f"{trigger_sigma}σ", color='blue', va='center', fontsize=10)
    ax.text(x_side, -trigger_thr, f"-{trigger_sigma}σ", color='blue', va='center', fontsize=10)
    ax.text(x_side, +boundary_thr, f"{boundary_sigma}σ", color='green', va='center', fontsize=10)
    ax.text(x_side, -boundary_thr, f"-{boundary_sigma}σ", color='green', va='center', fontsize=10)

    # Highlight each spark and mark peak
    for s in sparks.values():
        ax.axvspan(s['t_start'], s['t_end'], color='orange', alpha=0.3)
        ax.scatter(s['peak_time'], s['peak_adc'], color='red', s=50, zorder=5)

    ax.set_xlabel("Timestamp (µs)")
    ax.set_ylabel("ADC value")
    ax.set_title("Full Signal with Abnormal Events")
    plt.tight_layout()
    fig.show() if not save_outputs else None
    save_plot(fig, "full_signal_sparks", adc_sparks_dir, save_outputs)

    #=================== Voltage Plot ===================
    fig, ax = plt.subplots(figsize=(16,4), dpi=100)

    # Plot full signal in voltage
    adc_voltage = df['adc_value'] * config['adc_to_voltage']
    ax.plot(df['timestamp'], adc_voltage, lw=0.5, label='Signal (µV)')

    # Plot range-specific thresholds in voltage
    for start_idx, end_idx, local_std in noise_ranges:
        t_start = df['timestamp'].iloc[start_idx]
        t_end = df['timestamp'].iloc[end_idx]

        trigger_thr_v = trigger_sigma * local_std * config['adc_to_voltage']
        boundary_thr_v = boundary_sigma * local_std * config['adc_to_voltage']

        ax.hlines([+trigger_thr_v, -trigger_thr_v], t_start, t_end, color='blue', linestyle='--', lw=3, alpha=1)
        ax.hlines([+boundary_thr_v, -boundary_thr_v], t_start, t_end, color='green', linestyle=':', lw=3, alpha=1)

    # Side labels outside the plot
    x_side = df['timestamp'].iloc[-1] + 0.01*(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])
    ax.text(x_side, +trigger_thr_v, f"{trigger_sigma}σ", color='blue', va='center', fontsize=10)
    ax.text(x_side, -trigger_thr_v, f"-{trigger_sigma}σ", color='blue', va='center', fontsize=10)
    ax.text(x_side, +boundary_thr_v, f"{boundary_sigma}σ", color='green', va='center', fontsize=10)
    ax.text(x_side, -boundary_thr_v, f"-{boundary_sigma}σ", color='green', va='center', fontsize=10)

    # Highlight sparks and mark peaks
    for s in sparks.values():
        ax.axvspan(s['t_start'], s['t_end'], color='orange', alpha=0.3)
        ax.scatter(s['peak_time'], s['peak_adc'] * config['adc_to_voltage'], color='red', s=50, zorder=5)

    ax.set_xlabel("Timestamp (µs)")
    ax.set_ylabel("Voltage (µV)")
    ax.set_title("Full Signal with Abnormal Events")
    plt.tight_layout()
    fig.show() if not save_outputs else None
    save_plot(fig, "full_signal_sparks_voltage", voltage_sparks_dir, save_outputs)

# %%
# do fft, here i give it the adc_to_voltage conversion factor, so the fft is computed from the voltage signal, (if uV if i have set the conversion as such)
fft_df, f_nyquist,delta_f = fft_dataframe(df, scale=config['adc_to_voltage'])
print("=============== FFT Dataframe ===============")
print(fft_df.head())
print(fft_df.tail())
print("=============================================\n")

# %%
# plot the fft dataframe

fig, ax = plt.subplots(figsize=(16,4), dpi=100)
ax.plot(fft_df["frequency"], fft_df["magnitude"])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Fourier transform of the run")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "fft", fft_dir, save_outputs)
# %% find the 100 top peaks, just for filing
peaks = detect_fft_peaks(fft_df, neighborhood=100, top_n=100)
print("=============== Top 100 Peaks ===============")

# Print header
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
# Print rows with index starting from 1
for i, row in enumerate(peaks.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")

print("=========================================\n")

#%% find the 10 top peaks, needle like

peaks = detect_fft_peaks(fft_df, neighborhood=100, top_n=10)
print("=============== Top 10 Peaks ===============")

# Print header
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
# Print rows with index starting from 1
for i, row in enumerate(peaks.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")

print("=========================================\n")


#%%
fig, ax = plot_with_peaks(fft_df, peaks)
fig.show() if not save_outputs else None
save_plot(fig, "fft_with_peaks", fft_dir, save_outputs)

# here the interactive plots dont work because if we keep all points to see the peaks, it crashes, and if we downsample it drops the peaks
#interactive_plot(fft_df, downsample = 1000, extra_scatter={"x": peaks["frequency"], "y": peaks["magnitude"], "name": "Peaks"}) if not save_outputs else None

# %% 0-1kHz

fig, ax = plot_df_range(fft_df, x_col="frequency", y_col="magnitude", x_min=0, x_max=1000,
              title="FFT (0–1 kHz)", xlabel="Frequency [Hz]", ylabel="Magnitude")

fig.show() if not save_outputs else None
save_plot(fig,"User_Range_fft_plot", fft_dir, save_outputs)

peaks = detect_fft_peaks(fft_df.loc[fft_df["frequency"].between(0, 1000)], neighborhood=100, top_n=10)

print("=============== Top 10 Peaks, 0-1kHz range ===============")
# Print header
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
# Print rows with index starting from 1
for i, row in enumerate(peaks.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")

print("==========================================================\n")

fig, ax = plot_with_peaks(fft_df.loc[fft_df["frequency"].between(0, 1000)], peaks)
fig.show() if not save_outputs else None
save_plot(fig, "User_Range_with_peaks", fft_dir, True)

# %%
#region rebinned
#rebin the fft by bin width

fft_df_rebinned = rebin_by_width_centered_power_preserved_2(fft_df,config['FFT_rebin_width'])
fig, ax = plt.subplots(figsize=(16,4))
ax.plot(fft_df_rebinned["frequency"], fft_df_rebinned["magnitude"])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude")
ax.set_title(f"Rebinned FFT Dataframe ({config['FFT_rebin_width']} Hz bins)")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"rebinned_fft_plot", fft_dir, save_outputs)

fft_save_path = Path(fft_dir) / "fft_dataframe.parquet"
fft_df_rebinned.to_parquet(fft_save_path, index=False)
print(f"Saved FFT dataframe to: {fft_save_path}")
# %%
#plot peaks of rebinned
# here the new function to detect peaks doesnt work well, will have to look into that
peaks_rebinned = detect_fft_peaks(fft_df_rebinned,neighborhood=10, top_n=10)
print("=============== Top Peaks (Rebinned) ===============")
print(f"New bin width (in Hz):\t{config['FFT_rebin_width']}")

# Print header
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
# Print rows with index starting from 1
for i, row in enumerate(peaks_rebinned.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")

print("====================================================\n")
fig, ax = plot_with_peaks(fft_df_rebinned,peaks_rebinned)
fig.show() if not save_outputs else None
save_plot(fig,"rebinned_fft_with_peaks", fft_dir, save_outputs)
#endregion


# %% 0-1kHz, rebinned

fig, ax = plot_df_range(fft_df_rebinned, x_col="frequency", y_col="magnitude", x_min=0, x_max=1000,
              title="FFT (0–1 kHz)", xlabel="Frequency [Hz]", ylabel="Magnitude")

fig.show() if not save_outputs else None
save_plot(fig,"User_Range_fft_plot_rebinned", fft_dir, save_outputs)

peaks = detect_fft_peaks(fft_df_rebinned.loc[fft_df_rebinned["frequency"].between(0, 1000)], neighborhood=1, top_n=10)

print("=============== Top 10 Peaks, 0-1kHz range, rebinned ===============")
# Print header
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
# Print rows with index starting from 1
for i, row in enumerate(peaks.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")

print("====================================================================\n")

fig, ax = plot_with_peaks(fft_df_rebinned.loc[fft_df_rebinned["frequency"].between(0, 1000)], peaks)
fig.show() if not save_outputs else None
save_plot(fig, "User_Range_with_peaks_rebinned", fft_dir, save_outputs)

#%%
# interactive_plot(fft_df_rebinned, downsample=1)
# %%
# calculate noise metrics from fft dataframe
metrics_df = calculate_noise_metrics_from_single_df(fft_df)
print("=============== Noise Metrics ===============") 
print("Low: 0-1 kHz\tMid: 1-10 kHz\tHigh: >10 kHz\n")
for col in metrics_df.columns:
    print(f"{col}: {metrics_df[col][0]:.2f}")
print(f"Nyquist frequency (Hz): {f_nyquist:.2f}")
print(f"Max Frequency resolution (Hz): {delta_f:.2g}")
print("=============================================\n")   




# %%
