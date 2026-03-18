#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import matplotlib.pyplot as plt
import os
from pathlib import Path

#%%
# === Plot Style Config ===
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%% 
# Load the config file
with open("config_2.json", "r") as f:
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

#%% File conversion and import to dataframe.
if info["event_binary"] :   # event_binary is 1 for binary
    df = process_binary_to_parquet_3(emd_files[0], count_to=config["entry_limit"], verbose=True)
else :                      # if its 0 then its txt
    df = read_ascii_emd(emd_files[0])

print(df.head())
print(df.tail())
#print(df.to_string())


#%% Assign id numbers to the measurements, and removing the marker rows

split_measurements_inplace(df, adc_threshold=10000)

#%% Inspect if the split was successful

# get a measurement 
example_meas = df[df["meas_id"] == 1]

# plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(example_meas["timestamp"], example_meas["adc_value"])
ax.set_xlabel("Timestamp [µs]")
ax.set_ylabel("ADC Value")
ax.set_title("Individual Measurement")
plt.show()


#%% Shift each measurement timestamp to start from zero

shift_measurements_timestamps(df)

#%% Inspect 

# get a measurement 
example_meas = df[df["meas_id"] == 1]

# plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(example_meas["timestamp"], example_meas["adc_value"])
ax.set_xlabel("Timestamp [µs]")
ax.set_ylabel("ADC Value")
ax.set_title("Individual Measurement, timestamp shifted")
plt.show()


#%% Make a dataframe with the peak timestamps for each measurement

peak_df = get_two_peaks_per_measurement(df)
print(peak_df.head())

#%% Make a histogram of the values of the peak_df

# set min and max for histogram
min_time = 0
max_time = 120000  # microseconds, for example

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(peak_df["peak_timestamp"], bins=50, range=(min_time, max_time))
        #,
        #color="#4c72b0", edgecolor="black")
ax.set_xlabel("Timestamp of Peak ADC [µs]")
ax.set_ylabel("Counts")
ax.set_title("Histogram of Peak Point Timestamps per Measurement")
plt.tight_layout()
plt.show()




# %%
