#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sounddevice as sd


#%% Configuring plot style
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
fig.show()


print(df.head())

#%%
# test filter_df function
df = filter_df(df, 0, 600)  # keeps timestamps between the two time values in seconds
print(df.head())

# %%
# calculate the baseline using some range
# the range is declared in respect of the percentage of the full range. Can put any number of entries in ranges list

ranges = [
    (0.00, 0.25),
    (0.25, 0.75),
    (0.75, 1.0)
]

baseline, std_val = baseline_shift(df, ranges)
print("Baseline:", baseline)
print("Standard Deviation: ",std_val)

#%%
# plot the shifted dataframe

fig, ax = plt.subplots(figsize=(6,4), dpi=100)
ax.plot(df["timestamp"], df["adc_value"])
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Shifted by baseline")
plt.tight_layout()
fig.show()




# %%
time_us = df["timestamp"].to_numpy()
adc = df["adc_value"].to_numpy()

fs = int(1e6 / np.mean(np.diff(time_us)))  # <- make it int

signal = adc.astype(float)
signal -= np.mean(signal)
signal /= np.max(np.abs(signal))

sd.play(signal, fs)
sd.wait()

from scipy.io.wavfile import write
signal16 = np.int16(signal / np.max(np.abs(signal)) * 32767)

write("new_detector_music.wav", fs, signal16)
# %%

# # FFT
# fft_vals = np.fft.fft(signal)
# freqs = np.fft.fftfreq(len(signal), d=1/fs)  # Hz

# # Choose cutoff frequency (tweak this)
# f_cut = 4000  # Hz, remove above 10 kHz

# # Zero out high-frequency components
# fft_filtered = fft_vals.copy()
# fft_filtered[np.abs(freqs) > f_cut] = 0

# # inverse FFT
# signal_filtered = np.fft.ifft(fft_filtered).real

# # renormalize
# signal_filtered -= np.mean(signal_filtered)
# signal_filtered /= np.max(np.abs(signal_filtered))

#%%
# FFT
fft_vals = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), d=1/fs)

# --------------------------
# Step 1: Remove high frequencies above cutoff
# --------------------------
f_cut = 4000  # Hz
fft_high_removed = fft_vals.copy()
fft_high_removed[np.abs(freqs) >= f_cut] = 0

signal_high_removed = np.fft.ifft(fft_high_removed).real
signal_high_removed -= np.mean(signal_high_removed)
signal_high_removed /= np.max(np.abs(signal_high_removed))

# Play and save first filtered version
print("Playing: high-frequency cutoff only...")
sd.play(signal_high_removed, fs)
sd.wait()

write("new_detector_music_highcut.wav", fs, np.int16(signal_high_removed * 32767))


# %%
sd.play(signal_filtered, fs)
sd.wait()
# %%
