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



#%%
# filter_df does two things, it first shifts the timestamps to start from 0 and it filters the values to be up to a limit
df = filter_df(df, 0, config['timestamp_lim'])  # keeps timestamps between the two time values in seconds
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df.head())
print(df.tail())
print("=============================================\n")





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



#%% Plot around a specific timestamp
center_time = 593483697.00  # µs, replace with your value
time_range  = 42768.30   # µs, total window around the center

# Define start and end of the window
start_time = center_time - time_range/2
end_time   = center_time + time_range/2

# Slice the dataframe
df_window = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

# Plot
fig, ax = plt.subplots(figsize=(12,4), dpi=100)
ax.plot(df_window["timestamp"], df_window["adc_value"]*config["adc_to_voltage"], lw=1)
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("Voltage (μV)")
ax.set_title(f"Signal from {start_time:.0f} to {end_time:.0f} µs")

plt.tight_layout()
fig.show() if not save_outputs else None



# %%
parquet_file = "/home/esamouil/data_ess__/data_psi_aug_2024/DREAM_B/Z010824_0002/baseline_corrected_first10.parquet"

# Load the dataframe
df_parquet = pd.read_parquet(parquet_file)

# Plot
fig, ax = plt.subplots(figsize=(12,4), dpi=100)
ax.plot(df_parquet["timestamp"], df_parquet["adc_value"]*config["adc_to_voltage"], lw=1)
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC Value")
ax.set_title("ADC Signal from Parquet")
plt.tight_layout()
fig.show()

#%%
# Shift amount in µs
i = 593483697.00 - 600000

# Create the shifted Parquet dataframe
df_parquet_shifted = df_parquet.copy()
df_parquet_shifted["timestamp"] = df_parquet_shifted["timestamp"] + i

# Full range of the Parquet dataframe
start_time = df_parquet_shifted["timestamp"].min()
end_time   = df_parquet_shifted["timestamp"].max()

# Slice EMD within that range
df_window_emd = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

# Plot
fig, ax = plt.subplots(figsize=(12,4), dpi=100)
ax.plot(df_parquet_shifted["timestamp"], df_parquet_shifted["adc_value"]*config["adc_to_voltage"], lw=1, label=f"Beam Signal")
ax.plot(df_window_emd["timestamp"], df_window_emd["adc_value"]*config["adc_to_voltage"], lw=1, label="Abnormal Event 17")
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("Voltage (μV)")
ax.set_title("Abnormal Event and Beam Signal Comparison")
ax.legend()
plt.tight_layout()
fig.show() if not save_outputs else None

# %%
