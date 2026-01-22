#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import matplotlib.pyplot as plt


#%%
# === Plot Style Config ===
plt.rcParams.update({
    # Font
    "font.family": "Nimbus Roman",
    "mathtext.rm": "Nimbus Roman",
    "font.size": 14,
    # Figure
    "figure.figsize": (6, 4),
    "figure.dpi": 100,
    # Lines and markers
    "lines.linewidth": 2,
    "lines.markersize": 6,
    # Axes labels and ticks
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # Grid
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.alpha": 0.7,
    # Legend
    "legend.fontsize": 12,
    # Error bars
    "errorbar.capsize": 4,
    # Boxplots
    "boxplot.flierprops.markersize": 4,
    "boxplot.meanprops.markersize": 4,
    #colour palette
    "axes.prop_cycle": plt.cycler(color=["#4c72b0",
     "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
)
})



#%% 
# Load the config file
with open("config.json", "r") as f:
    config = json.load(f)

#%%
#find the files 

# Folder to scan
#folder_path = "/home/esamouil/analysis/my_scripts/analysis_files"

# Convert to Path object
folder = Path(config["data_folder_path"])

# Get all .emd and .txt files
emd_files = list(folder.glob("*.emd"))
txt_files = list(folder.glob("*.txt"))

print("Found EMD files:", [f.name for f in emd_files])
print("Found TXT files:", [f.name for f in txt_files])    

#%%
# Get parameters from the txt file.
info = parse_txt_file(txt_files[0])
print(info)

    
#%% File conversion and import to dataframe.

if info["event_binary"] :   # event_binary is 1 for binary
    df = process_binary_to_parquet(emd_files[0], count_to=config["entry_limit"], verbose=True)
else :                      # if its 0 then its txt
    df = read_ascii_emd(emd_files[0])

print(df.head())
print(df.tail())
# %%
# plot the dataframe

plt.plot(df["timestamp"], df["adc_value"])
plt.xlabel("Timestamp (µs)")
plt.ylabel("ADC value")
plt.title("Test title")

# Little info box in top-right corner
textstr = f'Detector: {info["detector_name"]}'

plt.gca().text(
    0.35, 0.95, textstr, transform=plt.gca().transAxes,
    fontsize=12, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
)

plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()
print(df.head())

#%%
# test filter_df function
df = filter_df(df, 0, 59)  # keeps timestamps between 0.5s and 2.0s
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

plt.plot(df["timestamp"], df["adc_value"])
plt.xlabel("Timestamp (µs)")
plt.ylabel("ADC value")
plt.tight_layout()
plt.show()
# %%
# do fft
fft_df = fft_dataframe(df)
print(fft_df.head())

# %%
# plot the fft dataframe

plt.plot(fft_df["frequency"], fft_df["magnitude"])
plt.xlabel("frequency (HZ)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.show()


# %%
peaks = find_top_peaks(fft_df, n=10)
print(peaks)

#%%
plot_with_peaks(fft_df, peaks)

# %%
plot_df_range(fft_df, x_col="frequency", y_col="magnitude", x_min=0, x_max=1000,
              title="FFT (0–1 kHz)", xlabel="Frequency [Hz]", ylabel="Amplitude")
# %%
#rebin the adc signal

fft_df_rebinned = rebin_by_width(fft_df,50)
plt.plot(fft_df_rebinned["frequency"], fft_df_rebinned["magnitude"])
plt.xlabel("frequency (Hz)")
plt.ylabel("magnitude")
plt.tight_layout()
plt.show()
# %%
#plot peaks of rebinned
peaks_rebinned = find_top_peaks(fft_df_rebinned, n=10)
plot_with_peaks(fft_df_rebinned,peaks_rebinned)
# %%
# calculate noise metrics from fft dataframe
metrics_df = calculate_noise_metrics_from_single_df(fft_df)
print(metrics_df)





# %%
