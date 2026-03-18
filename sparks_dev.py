#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%% Load config
with open("config_sparks.json", "r") as f:
    config = json.load(f)

folder = Path(config["data_folder_path"])
txt_file = folder / config["single_file_name"]
stem = txt_file.stem
emd_file = list(folder.glob(f"{stem}*.emd"))[0]

print("EMD:", emd_file.name)
print("TXT:", txt_file.name)

#%% Parse run info
info = parse_txt_file(txt_file)
print(info)

#%% Load data
if info["event_binary"]:
    df = process_binary_to_parquet_3(
        emd_file,
        count_to=config["entry_limit"],
        verbose=True
    )
else:
    df = read_ascii_emd(emd_file)

#%% Timestamp filtering
df = filter_df(df, 0, config["timestamp_lim"])

#%% Baseline + noise
ranges = [(0.0, 1.0)]
baseline, std_val = baseline_shift(df, ranges)

print("Baseline:", baseline)
print("Std:", std_val)

# ============================================================
# SPARK DETECTION LOGIC
# ============================================================

trigger_sigma = 6
boundary_sigma = 4      # for stable region
time_window = 10000     # µs, max separation for cluster
min_points = 3          # min full-resolution points to be considered spark
t_stable = 20000        # µs, stable region duration before/after spark
zoom_frac = 0.5         # zoom 50% before/after duration

threshold_trigger = trigger_sigma * std_val
threshold_boundary = boundary_sigma * std_val

# Convert to numpy arrays
ts = df["timestamp"].to_numpy()
adc = df["adc_value"].to_numpy()
over_trigger = np.abs(adc) > threshold_trigger
over_boundary = np.abs(adc) > threshold_boundary

# Identify indices above trigger
flagged_indices = np.where(over_trigger)[0]

if len(flagged_indices) == 0:
    print("No sparks detected.")
    clusters = []
else:
    # Build clusters
    clusters = []
    current_cluster = [flagged_indices[0]]
    for idx in flagged_indices[1:]:
        dt = ts[idx] - ts[current_cluster[-1]]
        if dt <= time_window:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)

    # Keep clusters with enough points
    clusters = [c for c in clusters if len(c) >= min_points]

print(f"Detected sparks: {len(clusters)}")

# ============================================================
# Vectorized stable-region duration calculation
# ============================================================

# Precompute indices for blocks
ts_int = ts.astype(int)
stable_before = np.zeros_like(adc, dtype=bool)
stable_after  = np.zeros_like(adc, dtype=bool)

# Compute running min/max in blocks of t_stable
# Convert to index-based for efficiency
ts_sorted_idx = np.arange(len(ts))

for i in range(len(ts)):
    start_idx = np.searchsorted(ts, ts[i]-t_stable)
    stable_before[i] = np.all(~over_boundary[start_idx:i+1])
    end_idx   = np.searchsorted(ts, ts[i]+t_stable)
    stable_after[i] = np.all(~over_boundary[i:end_idx])

# ============================================================
# Extract spark peak and duration
# ============================================================

spark_peaks = []

for i, cluster in enumerate(clusters):
    cluster_ts = ts[cluster]
    cluster_adc = adc[cluster]

    # Peak
    peak_idx = cluster[np.argmax(np.abs(cluster_adc))]
    peak_time = ts[peak_idx]
    peak_adc_val = adc[peak_idx]

    # Vectorized t_start
    idx_before = cluster[0]
    stable_candidates = np.where(stable_before[:idx_before])[0]
    t_start = ts[stable_candidates[-1]] if len(stable_candidates) > 0 else ts[0]

    # Vectorized t_end
    idx_after = cluster[-1]
    stable_candidates = np.where(stable_after[idx_after:])[0]
    t_end = ts[idx_after + stable_candidates[0]] if len(stable_candidates) > 0 else ts[-1]

    duration = max(0, t_end - t_start)

    spark_peaks.append({
        "cluster_id": i+1,
        "t_start": t_start,
        "t_end": t_end,
        "duration": duration,
        "peak_time": peak_time,
        "peak_adc": peak_adc_val,
        "n_points": len(cluster)
    })

# Print results
for s in spark_peaks:
    print(
        f"Spark {s['cluster_id']} | "
        f"Points: {s['n_points']} | "
        f"Duration: {s['duration']} µs | "
        f"Peak @ {s['peak_time']} µs | "
        f"ADC {s['peak_adc']}"
    )

# ============================================================
# Plot full signal with spark peaks
# ============================================================

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ts,
    y=adc,
    mode="lines",
    name="ADC"
))

for s in spark_peaks:
    fig.add_trace(go.Scatter(
        x=[s["peak_time"]],
        y=[s["peak_adc"]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="x"),
        name=f"Spark {s['cluster_id']}"
    ))

fig.update_layout(
    title="ADC Signal with Detected Sparks",
    xaxis_title="Timestamp (µs)",
    yaxis_title="ADC value"
)
fig.show()

# ============================================================
# Zoomed plots with highlighted spark duration and thresholds
# ============================================================

for s in spark_peaks:
    zoom_before = zoom_frac * s["duration"]
    zoom_after  = zoom_frac * s["duration"]

    t_min = max(0, s["t_start"] - zoom_before)
    t_max = min(ts[-1], s["t_end"] + zoom_after)

    mask_zoom = (ts >= t_min) & (ts <= t_max)
    ts_zoom = ts[mask_zoom]
    adc_zoom = adc[mask_zoom]

    if len(ts_zoom) == 0:
        continue  # skip empty zoom

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_zoom,
        y=adc_zoom,
        mode="lines",
        name="ADC"
    ))

    # Highlight spark duration
    fig.add_trace(go.Scatter(
        x=[s["t_start"], s["t_end"], s["t_end"], s["t_start"], s["t_start"]],
        y=[adc_zoom.min()]*2 + [adc_zoom.max()]*2 + [adc_zoom.min()],
        fill="toself",
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,0,0,0)"),
        name="Spark duration"
    ))

    # Peak marker
    fig.add_trace(go.Scatter(
        x=[s["peak_time"]],
        y=[s["peak_adc"]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="x"),
        name="Peak"
    ))

    # Threshold lines
    fig.add_hline(y=threshold_trigger, line=dict(color="blue", dash="dash"), annotation_text="6σ", annotation_position="top left")
    fig.add_hline(y=-threshold_trigger, line=dict(color="blue", dash="dash"), annotation_text="-6σ", annotation_position="bottom left")
    fig.add_hline(y=threshold_boundary, line=dict(color="green", dash="dot"), annotation_text=f"{boundary_sigma}σ", annotation_position="top right")
    fig.add_hline(y=-threshold_boundary, line=dict(color="green", dash="dot"), annotation_text=f"-{boundary_sigma}σ", annotation_position="bottom right")

    fig.update_layout(
        title=f"Spark {s['cluster_id']} (Zoomed)",
        xaxis_title="Timestamp (µs)",
        yaxis_title="ADC value"
    )
    fig.show()

# %%
