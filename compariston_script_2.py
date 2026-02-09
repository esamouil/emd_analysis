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
with open("config_comp.json", "r") as f:
    config = json.load(f)
save_outputs = config.get("save_outputs", False)

#%%
# Convert to Path object
folder = Path(config["data_folder_path"])
txt_files_1 = [folder / config["single_file_name_1"]]
txt_files_2 = [folder / config["single_file_name_2"]]
# find matching .emd file(s) by some convention, e.g. same stem
stem_1 = txt_files_1[0].stem
stem_2 = txt_files_2[0].stem
emd_files_1 = list(folder.glob(f"{stem_1}*.emd"))
emd_files_2 = list(folder.glob(f"{stem_2}*.emd"))

#%% Saving outputs setup
output_dir = os.path.join(config["data_folder_path"], "comparison")

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
print("EMD File:", [f.name for f in emd_files_1])
print("TXT File:", [f.name for f in txt_files_1])
print("===================================\n")  
print("=============== Run ===============")
print("EMD File:", [f.name for f in emd_files_2])
print("TXT File:", [f.name for f in txt_files_2])
print("===================================\n")   

#%%
# Get parameters from the txt file.
info_1 = parse_txt_file(txt_files_1[0])
print("=============== Run Info ===============")
for k, v in info_1.items():
    print(f"{k}: {v}")
print("========================================\n")

info_2 = parse_txt_file(txt_files_2[0])
print("=============== Run Info ===============")
for k, v in info_2.items():
    print(f"{k}: {v}")
print("========================================\n")

#%% Get info from the metadata excel

# Load the Excel with the info
excel_path = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/logbook_Z_with_baselines.xlsx"  # replace with the actual path
excel_df = pd.read_excel(excel_path)

# Create a lookup dictionary: {filename: (actual_detector, boolean)}
file_info_1 = {
    row['filename']: (row['act.det'], bool(row['corr. needed']))
    for _, row in excel_df.iterrows()
}

file_info_2 = {
    row['filename']: (row['act.det'], bool(row['corr. needed']))
    for _, row in excel_df.iterrows()
}

# Example: get the info for the current txt file
current_file_1 = txt_files_1[0].name
if current_file_1 in file_info_1:
    actual_detector_1, cor_boolean_1 = file_info_1[current_file_1]
    print(f"File: {current_file_1}")
    print(f"Actual Detector: {actual_detector_1}")
    print(f"Correction Boolean: {cor_boolean_1}")
else:
    print(f"No entry for {current_file_1} in Excel")

# Example: get the info for the current txt file
current_file_2 = txt_files_2[0].name
if current_file_2 in file_info_2:
    actual_detector_2, cor_boolean_2 = file_info_2[current_file_2]
    print(f"File: {current_file_2}")
    print(f"Actual Detector: {actual_detector_2}")
    print(f"Correction Boolean: {cor_boolean_2}")
else:
    print(f"No entry for {current_file_2} in Excel")

#%% File conversion and import to dataframe.
print("=============== Data Import ===============")
if info_1["event_binary"] :   # event_binary is 1 for binary
    df_1 = process_binary_to_parquet_3(emd_files_1[0], count_to=config["entry_limit"], verbose=True)
else :                      # if its 0 then its txt
    df_1 = read_ascii_emd(emd_files_1[0])

print(df_1.head())
print(df_1.tail())
print("========================================\n")

print("=============== Data Import ===============")
if info_2["event_binary"] :   # event_binary is 1 for binary
    df_2 = process_binary_to_parquet_3(emd_files_2[0], count_to=config["entry_limit"], verbose=True)
else :                      # if its 0 then its txt
    df_2 = read_ascii_emd(emd_files_2[0])

print(df_2.head())
print(df_2.tail())
print("========================================\n")

# %%
# plot the dataframe

fig, ax = plt.subplots(figsize=(16,4), dpi=100)

ax.plot(df_1["timestamp"], df_1["adc_value"], label=Path(config['single_file_name_1']).stem,alpha=0.7)
ax.plot(df_2["timestamp"], df_2["adc_value"], label=Path(config['single_file_name_2']).stem,alpha=0.7)

ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Raw file Comparison")

# Little info box with detectors
textstr = f'Detector 1: {info_1["detector_name"]}\nDetector 2: {info_2["detector_name"]}'
ax.text(
    0.35, 0.95, textstr, transform=ax.transAxes,
    fontsize=12, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
)

ax.legend()
fig.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "raw_file_comparison", output_dir, save_outputs)


#%%
# filter_df does two things, it first shirt the timestamps to start from 0 and it filters the values to be up to a limit
df_1 = filter_df(df_1, 0, config['timestamp_lim'])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df_1.head())
print(df_1.tail())
print("=============================================\n")

# filter_df does two things, it first shirt the timestamps to start from 0 and it filters the values to be up to a limit
df_2 = filter_df(df_2, 0, config['timestamp_lim'])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df_2.head())
print(df_2.tail())
print("=============================================\n")

#%% Baseline and std calculation before the TIA sampling point correction

baseline_1 = df_1["adc_value"].mean()
std_val_1  = df_1["adc_value"].std()

print("=============== Pre TIA samp. Correction Baseline ===============")
print("Pre_Correction_Baseline:", baseline_1)
print("Pre_Correction_Standard deviation:", std_val_1)
print("=================================================================\n")

baseline_2 = df_2["adc_value"].mean()
std_val_2  = df_2["adc_value"].std()

print("=============== Pre TIA samp. Correction Baseline ===============")
print("Pre_Correction_Baseline:", baseline_2)
print("Pre_Correction_Standard deviation:", std_val_2)
print("=================================================================\n")


#%% Normalize based on correction scale factor and TIA points

# scale adc values
scale_factor_1 = info_1["tia_summation_points"]
if cor_boolean_1:  # from Excel lookup
    scale_factor_1 *= 4

df_1["adc_value"] = df_1["adc_value"] / scale_factor_1

print(f"ADC values scaled by {scale_factor_1}")

print(df_1.head())
print(df_1.tail())

# scale adc values
scale_factor_2 = info_2["tia_summation_points"]
if cor_boolean_2:  # from Excel lookup
    scale_factor_2 *= 4

df_2["adc_value"] = df_2["adc_value"] / scale_factor_2

print(f"ADC values scaled by {scale_factor_2}")

print(df_2.head())
print(df_2.tail())

# %%
# calculate the baseline using some range
# the range is declared in respect of the percentage of the full range. Can put any number of entries in ranges list

# ranges = [
#     (0.00, 0.25),
#     (0.25, 0.75),
#     (0.75, 1.0)
# ]
ranges =[(0.0,1.0)]

baseline_1, std_val_1 = baseline_shift(df_1, ranges)
print("=============== Baseline Calculation ===============")
print("Ranges (Percentage of range):")
for r in ranges:
    print(f"{r[0]*100} - {r[1]*100}")
print("Baseline:", baseline_1)
print("Standard Deviation: ",std_val_1)
print("===================================================\n")

baseline_2, std_val_2 = baseline_shift(df_2, ranges)
print("=============== Baseline Calculation ===============")
print("Ranges (Percentage of range):")
for r in ranges:
    print(f"{r[0]*100} - {r[1]*100}")
print("Baseline:", baseline_2)
print("Standard Deviation: ",std_val_2)
print("===================================================\n")


#%%
# plot the shifted dataframe

fig, ax = plt.subplots(figsize=(16,4), dpi=100)

ax.plot(df_1["timestamp"], df_1["adc_value"], label=Path(config['single_file_name_1']).stem, alpha=0.7, lw=1)
ax.plot(df_2["timestamp"], df_2["adc_value"], label=Path(config['single_file_name_2']).stem, alpha=0.7, lw=1)

ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Shifted by baseline Comparison")

# Info box with detectors
textstr = f'Detector 1: {info_1["detector_name"]}\nDetector 2: {info_2["detector_name"]}'
ax.text(
    0.35, 0.95, textstr, transform=ax.transAxes,
    fontsize=12, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
)

ax.legend()
fig.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "shifted_by_baseline_comparison", output_dir, save_outputs)

# %%
# do fft
fft_df_1 = fft_dataframe(df_1)
print("=============== FFT Dataframe ===============")
print(fft_df_1.head())
print(fft_df_1.tail())
print("=============================================\n")

fft_df_2 = fft_dataframe(df_2)
print("=============== FFT Dataframe ===============")
print(fft_df_2.head())
print(fft_df_2.tail())
print("=============================================\n")

# %%
# plot the fft dataframe

fig, ax = plt.subplots(figsize=(16,4), dpi=100)

ax.plot(fft_df_1["frequency"], fft_df_1["magnitude"], label=Path(config['single_file_name_1']).stem, alpha=0.7, lw=1)
ax.plot(fft_df_2["frequency"], fft_df_2["magnitude"], label=Path(config['single_file_name_2']).stem, alpha=0.7, lw=1)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("FFT Dataframe Comparison")

ax.legend()
fig.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "fft_dataframe_comparison", output_dir, save_outputs)



# %%
# peaks = find_top_peaks(fft_df, n=10)
# print("=============== Top Peaks ===============")
# print(peaks)
# print("=========================================\n")

#%%
# fig, ax = plot_with_peaks(fft_df, peaks)
# fig.show() if not save_outputs else None
# save_plot(fig, "fft_with_peaks", output_dir, save_outputs)

# %%
fig, ax = plt.subplots(figsize=(16,4), dpi=100)

# Plot the 0–1 kHz range for both FFTs
ax.plot(fft_df_1["frequency"], fft_df_1["magnitude"], 
        label=Path(config['single_file_name_1']).stem, alpha=0.7, lw=1)
ax.plot(fft_df_2["frequency"], fft_df_2["magnitude"], 
        label=Path(config['single_file_name_2']).stem, alpha=0.7, lw=1)

ax.set_xlim(0, 1000)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.set_title("FFT (0–1 kHz) Comparison")
ax.legend()

fig.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "UserRange_fft_comparison", output_dir, save_outputs)

# %%
#rebin the fft by bin width

fft_df_rebinned_1 = rebin_by_width(fft_df_1,config['FFT_rebin_width'])
fft_df_rebinned_2 = rebin_by_width(fft_df_2,config['FFT_rebin_width'])


fig, ax = plt.subplots(figsize=(16,4))

ax.plot(fft_df_rebinned_1["frequency"], fft_df_rebinned_1["magnitude"], 
        label=Path(config['single_file_name_1']).stem, alpha=0.7, lw=1)
ax.plot(fft_df_rebinned_2["frequency"], fft_df_rebinned_2["magnitude"], 
        label=Path(config['single_file_name_2']).stem, alpha=0.7, lw=1)

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude")
ax.set_title(f"Rebinned FFT Dataframe ({config['FFT_rebin_width']} Hz bins) Comparison")
ax.legend()

plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "rebinned_fft_comparison", output_dir, save_outputs)

# %%
# #plot peaks of rebinned
# peaks_rebinned = find_top_peaks(fft_df_rebinned, n=10)
# print("=============== Top Peaks (Rebinned) ===============")
# print(f"New bin width (in Hz):\t{config['FFT_rebin_width']}")
# print(peaks_rebinned)
# print("====================================================\n")
# fig, ax = plot_with_peaks(fft_df_rebinned,peaks_rebinned)
# fig.show() if not save_outputs else None
# save_plot(fig,"rebinned_fft_with_peaks", output_dir, save_outputs)
# %%
# calculate noise metrics from fft dataframe
metrics_df_1 = calculate_noise_metrics_from_single_df(fft_df_1)
print("=============== Noise Metrics ===============") 
print("Low: 0-1 kHz\tMid: 1-10 kHz\tHigh: >10 kHz\n")
print(metrics_df_1)
print("=============================================\n")   

# calculate noise metrics from fft dataframe
metrics_df_2 = calculate_noise_metrics_from_single_df(fft_df_2)
print("=============== Noise Metrics ===============") 
print("Low: 0-1 kHz\tMid: 1-10 kHz\tHigh: >10 kHz\n")
print(metrics_df_2)
print("=============================================\n")  


# %%
