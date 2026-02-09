#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import os
from pathlib import Path


#%% Configuring plot style

# plt.rcParams.update({
#     # Font
#     "font.family": "Nimbus Roman",
#     "mathtext.rm": "Nimbus Roman",
#     "font.size": 14,
#     # Figure
#     "figure.figsize": (6, 4),
#     "figure.dpi": 100,
#     # Lines and markers
#     "lines.linewidth": 2,
#     "lines.markersize": 6,
#     # Axes labels and ticks
#     "axes.labelsize": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     # Grid
#     "axes.grid": True,
#     "grid.linestyle": "--",
#     "grid.color": "gray",
#     "grid.alpha": 0.7,
#     # Legend
#     "legend.fontsize": 12,
#     # Error bars
#     "errorbar.capsize": 4,
#     # Boxplots
#     "boxplot.flierprops.markersize": 4,
#     "boxplot.meanprops.markersize": 4,
#     #colour palette
#     "axes.prop_cycle": plt.cycler(color=["#4c72b0",
#      "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
# )
# })



plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')


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

#%% Get info from the metadata excel

# Load the Excel with the info
excel_path = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/logbook_Z_with_baselines.xlsx"  # replace with the actual path
excel_df = pd.read_excel(excel_path)

# Create a lookup dictionary: {filename: (actual_detector, boolean)}
file_info = {
    row['filename']: (row['act.det'], bool(row['corr. needed']))
    for _, row in excel_df.iterrows()
}

# Example: get the info for the current txt file
current_file = txt_files[0].name
if current_file in file_info:
    actual_detector, cor_boolean = file_info[current_file]
    print(f"File: {current_file}")
    print(f"Actual Detector: {actual_detector}")
    print(f"Correction Boolean: {cor_boolean}")
else:
    print(f"No entry for {current_file} in Excel")

#%% File conversion and import to dataframe.
print("=============== Data Import ===============")
if info["event_binary"] :   # event_binary is 1 for binary
    df = process_binary_to_parquet_3(emd_files[0], count_to=config["entry_limit"], verbose=True)
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
save_plot(fig, "raw_file", output_dir, save_outputs)


#%%
# filter_df does two things, it first shirt the timestamps to start from 0 and it filters the values to be up to a limit
df = filter_df(df, 0, config['timestamp_lim'])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df.head())
print(df.tail())
print("=============================================\n")

#%% Baseline and std calculation before the TIA sampling point correction

baseline = df["adc_value"].mean()
std_val  = df["adc_value"].std()

print("=============== Pre TIA samp. Correction Baseline ===============")
print("Pre_Correction_Baseline:", baseline)
print("Pre_Correction_Standard deviation:", std_val)
print("=================================================================\n")

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
ax.set_title("Last 2% of file (pre-TIA correction)")
ax.set_xlim(0, 120_000)   # <-- fixed axis

plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "last_2percent_pre_TIA_correction", output_dir, save_outputs)


#%% Normalize based on correction scale factor and TIA points

# scale adc values
scale_factor = info["tia_summation_points"]
if cor_boolean:  # from Excel lookup
    scale_factor *= 4

df["adc_value"] = df["adc_value"] / scale_factor

print(f"ADC values scaled by {scale_factor}")

print(df.head())
print(df.tail())

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
print("Baseline:", baseline)
print("Standard Deviation: ",std_val)
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
save_plot(fig,"shifted by baseline",output_dir, save_outputs)
# %%
# do fft
fft_df = fft_dataframe(df)
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
ax.set_title("FFT Dataframe")
plt.tight_layout()
fig.show() if not save_outputs else None


# %%
peaks = find_top_peaks(fft_df, n=10)
print("=============== Top Peaks ===============")
print(peaks)
print("=========================================\n")

#%%
fig, ax = plot_with_peaks(fft_df, peaks)
fig.show() if not save_outputs else None
save_plot(fig, "fft_with_peaks", output_dir, save_outputs)

# %%
fig, ax = plot_df_range(fft_df, x_col="frequency", y_col="magnitude", x_min=0, x_max=1000,
              title="FFT (0–1 kHz)", xlabel="Frequency [Hz]", ylabel="Amplitude")

fig.show() if not save_outputs else None
save_plot(fig,"User Range_fft_plot", output_dir, save_outputs)
# %%
#rebin the fft by bin width

fft_df_rebinned = rebin_by_width(fft_df,config['FFT_rebin_width'])
fig, ax = plt.subplots(figsize=(16,4))
ax.plot(fft_df_rebinned["frequency"], fft_df_rebinned["magnitude"])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude")
ax.set_title(f"Rebinned FFT Dataframe ({config['FFT_rebin_width']} Hz bins)")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"rebinned_fft_plot", output_dir, save_outputs)
# %%
#plot peaks of rebinned
peaks_rebinned = find_top_peaks(fft_df_rebinned, n=10)
print("=============== Top Peaks (Rebinned) ===============")
print(f"New bin width (in Hz):\t{config['FFT_rebin_width']}")
print(peaks_rebinned)
print("====================================================\n")
fig, ax = plot_with_peaks(fft_df_rebinned,peaks_rebinned)
fig.show() if not save_outputs else None
save_plot(fig,"rebinned_fft_with_peaks", output_dir, save_outputs)
# %%
# calculate noise metrics from fft dataframe
metrics_df = calculate_noise_metrics_from_single_df(fft_df)
print("=============== Noise Metrics ===============") 
print("Low: 0-1 kHz\tMid: 1-10 kHz\tHigh: >10 kHz\n")
print(metrics_df)
print("=============================================\n")   


# %%
