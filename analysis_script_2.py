#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import matplotlib.pyplot as plt
import os
from pathlib import Path

#%%
# # === Plot Style Config ===
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
with open("config_2.json", "r") as f:
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


#%% File conversion and import to dataframe.
print("=============== Data Import ===============")
if info["event_binary"] :   # event_binary is 1 for binary
    df = process_binary_to_parquet_3(emd_files[0], count_to=config["entry_limit"], verbose=True)
else :                      # if its 0 then its txt
    df = read_ascii_emd(emd_files[0])

print(df.head())
print(df.tail())
print("========================================\n")


#%% Assign id numbers to the measurements, and removing the marker rows

split_measurements_by_threshold(df, adc_threshold=10000)
print("=============== Tagging measurements ===============")
print(df.head())
print(df.tail())
print("====================================================\n")


#%% Inspect if the split was successful

# get a measurement 
example_meas = df[df["meas_id"] == 1]

# plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(example_meas["timestamp"], example_meas["adc_value"])
ax.set_xlabel("Timestamp [µs]")
ax.set_ylabel("ADC Value")
ax.set_title("Individual Measurement")
fig.show() if not save_outputs else None
save_plot(fig, "individual_measurement", output_dir, save_outputs)


#%% Shift each measurement timestamp to start from zero

shift_measurements_timestamps(df)
print("=============== Shifted measurement timestamps ===============")
print(df.head())
print(df.tail())
print("==============================================================\n")

#%% Inspect 

# get a measurement 
example_meas = df[df["meas_id"] == 1]

# plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(example_meas["timestamp"], example_meas["adc_value"])
ax.set_xlabel("Timestamp [µs]")
ax.set_ylabel("ADC Value")
ax.set_title("Individual Measurement, timestamp shifted")
fig.show() if not save_outputs else None
save_plot(fig, "individual_measurement_timestamp_shifted", output_dir, save_outputs)


#%% Make a dataframe with the peak timestamps for each measurement

peak_df = get_two_peaks_per_measurement(df)
print("=============== Timestamps of peak ADC values (two per measurement) ===============")
print(peak_df.head())
print(peak_df.tail())
print("===================================================================================\n")

#%% Save peak_df as ASCII 
if save_outputs:
    base_name = Path(config["single_file_name"]).stem  # remove extension
    peak_file = os.path.join(output_dir, f"{base_name}_ascii.txt")
    peak_df.to_csv(peak_file, sep="\t", index=False)
    print(f"Saved peak_df to {peak_file}")

#%% Make a histogram of the values of the peak_df

# set min and max for histogram
min_time = 0
max_time = 1e6*1.0/info["chopper_freq_hz"]  # microseconds, for example

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(peak_df["peak_timestamp"], bins=50, range=(min_time, max_time))
        #,
        #color="#4c72b0", edgecolor="black")
ax.set_xlabel("Timestamp of Peak ADC [µs]")
ax.set_ylabel("Counts")
ax.set_title("Histogram of Peak Point Timestamps per Measurement")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig, "histogram", output_dir, save_outputs)




# %%
