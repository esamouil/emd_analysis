#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly_pub_clean

#%% Plot style configuration
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%%
# For notebook rendering (if running in Jupyter)
import plotly.io as pio
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
stem = txt_files[0].stem
emd_files = list(folder.glob(f"{stem}*.emd"))


#%% Print the file names 
print("=============== Run ===============")
print("EMD File:", [f.name for f in emd_files])
print("TXT File:", [f.name for f in txt_files])
print("===================================\n")    

#%%
info = parse_txt_file(txt_files[0])
print("=============== Run Info ===============")
for k, v in info.items():
    print(f"{k}: {v}")
print("========================================\n")

#%% File conversion and import to dataframe.
print("=============== Data Import ===============")
if info["event_binary"]:
    df = process_binary_to_parquet_3(emd_files[0], count_to=config["entry_limit"], verbose=True)
else:
    df = read_ascii_emd(emd_files[0])

print(df.head())
print(df.tail())
print("========================================\n")

#%%
# filter and Downsample for plotting (1 in 1000 points)
t_min = 0   # µs
t_max = 6e7   # µs

df_range = df[(df["timestamp"] >= t_min) & (df["timestamp"] <= t_max)]

# Downsample for plotting (1 in 100 points)
df_sample = df_range.iloc[::1000, :]

# Interactive raw plot
fig = px.line(df_sample, x="timestamp", y="adc_value", title="Raw file",template="pub_clean")
fig.update_layout(
    xaxis_title="Timestamp (µs)",
    yaxis_title="ADC value",
)
fig.update_layout(width=1200, height=400)
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.95, y=0.95,
    text=f'Detector: {info["detector_name"]}',
    showarrow=False,
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    opacity=0.8
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "raw_file.html"))

#%%
df = filter_df(df, 0, 1e6)
df_sample = df.iloc[::1000, :]  # downsample filtered df
print("=============== Filtered Data ===============")
print(f"Timestamp Limit\t{config['timestamp_lim']}\tseconds")
print(df.head())
print(df.tail())
print("=============================================\n")

# %%
ranges = [(0.0,1.0)]
baseline, std_val = baseline_shift(df, ranges)
print("=============== Baseline Calculation ===============")
print("Ranges (Percentage of range):")
for r in ranges:
    print(f"{r[0]*100} - {r[1]*100}")
print("Baseline:", baseline)
print("Standard Deviation: ",std_val)
print("===================================================\n")

#%%
# Plot shifted dataframe (downsample)
df_sample = df.iloc[::1000, :]
fig = px.line(df_sample, x="timestamp", y="adc_value", title="Shifted by baseline",template="pub_clean")
fig.update_layout(
    xaxis_title="Timestamp (µs)",
    yaxis_title="ADC value",
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "shifted_by_baseline.html"))

# %%
fft_df, f_nyquist, delta_f = fft_dataframe(df)
print("=============== FFT Dataframe ===============")
print(fft_df.head())
print(fft_df.tail())
print("=============================================\n")

#%%
fft_sample = fft_df.iloc[::100, :]
fig = px.line(fft_sample, x="frequency", y="magnitude", title="FFT Dataframe",template="pub_clean")
fig.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "fft_dataframe.html"))

# %%
peaks = detect_fft_peaks(fft_df, neighborhood=1000, top_n=10)
print("=============== Top Peaks ===============")
print(peaks)
print("=========================================\n")

#%%
fig = px.line(fft_df, x="frequency", y="magnitude", title="FFT with Peaks")
fig.add_scatter(x=peaks['frequency'], y=peaks['magnitude'], mode='markers', name='Peaks')
fig.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "fft_with_peaks.html"))

# %%
fig = px.line(fft_sample, x="frequency", y="magnitude", title="FFT (0–1 kHz)")
fig.update_xaxes(range=[0,1000])
fig.update_layout(
    xaxis_title="Frequency [Hz]",
    yaxis_title="Amplitude",
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "User_Range_fft_plot.html"))

# %%
fft_df_rebinned = rebin_by_width(fft_df, 10)
fft_reb_sample = fft_df_rebinned.iloc[::1000, :]
fig = px.line(fft_reb_sample, x="frequency", y="magnitude", title=f"Rebinned FFT ({config['FFT_rebin_width']} Hz bins)")
fig.update_layout(
    xaxis_title="Frequency [Hz]",
    yaxis_title="Magnitude",
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "rebinned_fft_plot.html"))


# %%
peaks_rebinned = find_top_peaks_old(fft_df_rebinned, n=10)
print("=============== Top Peaks (Rebinned) ===============")
print(f"New bin width (in Hz):\t{config['FFT_rebin_width']}")
print(peaks_rebinned)
print("====================================================\n")

fig = px.line(fft_reb_sample, x="frequency", y="magnitude", title="Rebinned FFT with Peaks")
fig.add_scatter(x=peaks_rebinned['frequency'], y=peaks_rebinned['magnitude'], mode='markers', name='Peaks')
fig.update_layout(
    xaxis_title="Frequency [Hz]",
    yaxis_title="Magnitude",
)
fig.show()
if save_outputs:
    fig.write_html(os.path.join(output_dir, "rebinned_fft_with_peaks.html"))

# %%
metrics_df = calculate_noise_metrics_from_single_df(fft_df)
print("=============== Noise Metrics ===============") 
print("Low: 0-1 kHz\tMid: 1-10 kHz\tHigh: >10 kHz\n")
print(metrics_df)
print("=============================================\n")   

# %%
