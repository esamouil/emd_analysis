#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json 
import os
import plotly.express as px
import plotly.graph_objects as go

#%%
# === Plot Style Config ===

plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%%
# For notebook rendering (if running in Jupyter)
import plotly.io as pio
pio.renderers.default = "notebook"

#%% 
# Load the config file
with open("config.json", "r") as f:
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
#%%
# Downsample for plotting (1 in 1000 points)
df_sample = df.iloc[::50, :]

# Interactive raw plot
fig = px.line(df_sample, x="timestamp", y="adc_value", title="Raw file")
fig.update_layout(
    xaxis_title="Timestamp (µs)",
    yaxis_title="ADC value",
    width=1500,   # width in pixels
    height=500    # height in pixels
)
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



# %%
N_last = 100_000   # how many last points to inspect
step = 1          # downsample factor

# Take last N points, then downsample
df_sample = df.tail(N_last).iloc[::step, :]

# Interactive raw plot
fig = px.line(df_sample, x="timestamp", y="adc_value", title=f"Raw file (last {N_last} points)")
fig.update_layout(
    xaxis_title="Timestamp (µs)",
    yaxis_title="ADC value",
    width=1500,   # width in pixels
    height=500    # height in pixels
)

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

# %%
