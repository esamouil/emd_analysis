#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import os
from pathlib import Path

# This script is to try and average out the signal by splitting it into timewindows

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
with open("config_3.json", "r") as f:
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

# %%
# plot the dataframe

fig, ax = plt.subplots(figsize=(6,4), dpi=100)

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
df = filter_df(df, 0, config["timestamp_lim"])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df.head())
print(df.tail())
print("=============================================\n")

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

fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"])
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Shifted by baseline")
plt.tight_layout()
fig.show() if not save_outputs else None
save_plot(fig,"shifted by baseline",output_dir, save_outputs)


#%% Split df into 72 ms measurements
time_window = 72000

df = split_measurements_by_time(df, measurement_us=time_window)

# Check
print(df.head())
print(df.tail())
print("\nNumber of measurements:")
print(df['meas_id'].nunique())


# %% Shift the measurement timestamps
shift_measurements_timestamps(df)
print(df.head())
print(df.tail())

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






# %%
# Trim last measurement
last_meas_id = df["meas_id"].max()
df_trimmed = df[df["meas_id"] != last_meas_id]

# Number of measurements
num_meas = df_trimmed["meas_id"].nunique()

# Create new DataFrame with fixed timestamps
step = 10  # µs
timestamps = np.arange(0, time_window+1, step)
avg_adc = np.zeros_like(timestamps, dtype=float)

# Map each original timestamp to the nearest bin
orig_times = df_trimmed["timestamp"].values
orig_adc = df_trimmed["adc_value"].values

# Compute indices of the closest timestamps
indices = np.searchsorted(timestamps, orig_times)
# Correct indices where the previous timestamp is actually closer
indices = np.clip(indices, 1, len(timestamps)-1)
left_diff = orig_times - timestamps[indices-1]
right_diff = timestamps[indices] - orig_times
indices = np.where(left_diff < right_diff, indices-1, indices)

# Sum adc values into bins
np.add.at(avg_adc, indices, orig_adc)

# Divide by number of measurements to get the average
avg_adc /= num_meas

# Create DataFrame
avg_df = pd.DataFrame({"timestamp": timestamps, "adc_value": avg_adc})

# Plot to check
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(avg_df["timestamp"], avg_df["adc_value"])
ax.set_xlabel("Timestamp [µs]")
ax.set_ylabel("Average ADC Value")
ax.set_title("Averaged Measurement")
plt.show()
# %%
