#%%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from esamouil_functions import *
# --- Style ---
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%% -------- CONFIG --------

fft_files = [
    Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0014/fft/fft_dataframe.parquet"),
    Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0021/fft/fft_dataframe.parquet"),
    Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0020/fft/fft_dataframe.parquet"),
    Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0018/fft/fft_dataframe.parquet"),
]

labels = ["HV=0V", "HV=200V", "HV=800V", "HV=1500V"]

save_outputs = False
output_dir = Path("comparison_fft_plots")
output_dir.mkdir(exist_ok=True)

#%% -------- LOAD DATA --------

fft_dfs = [pd.read_parquet(f) for f in fft_files]

#%% -------- FULL RANGE PLOT --------

fig, ax = plt.subplots(figsize=(16,4), dpi=100)

for df, label in zip(fft_dfs, labels):
    ax.plot(df["frequency"], df["magnitude"], lw=1, label=label)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Rebinned FFT Comparison")
ax.legend()

plt.tight_layout()
plt.show() if not save_outputs else None

if save_outputs:
    fig.savefig(output_dir / "fft_comparison_full.png", dpi=300)

#%% -------- 0–1 kHz RANGE --------

fig, ax = plt.subplots(figsize=(16,4), dpi=100)

for df, label in zip(fft_dfs, labels):
    df_zoom = df[df["frequency"].between(0, 1000)]
    ax.plot(df_zoom["frequency"], df_zoom["magnitude"], lw=1, label=label)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Fourier Profile Range 0-1kHz")
ax.legend()

plt.tight_layout()
plt.show() if not save_outputs else None

if save_outputs:
    fig.savefig(output_dir / "fft_comparison_0_1kHz.png", dpi=300)

#%%
#%% -------- 2x2 GRID FULL RANGE --------

fig, axes = plt.subplots(2, 2, figsize=(14,8), dpi=100, sharex=True, sharey=True)

axes = axes.flatten()

for ax, df, label in zip(axes, fft_dfs, labels):
    ax.plot(df["frequency"], df["magnitude"], lw=1.2)
    ax.set_title(label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

fig.suptitle("Rebinned FFT Comparison (Full Range)", fontsize=14)

plt.tight_layout()
plt.show() if not save_outputs else None

if save_outputs:
    fig.savefig(output_dir / "fft_comparison_2x2_grid.png", dpi=300)
# %%
#%% -------- 2x2 GRID FULL RANGE WITH PEAKS --------

fig, axes = plt.subplots(2, 2, figsize=(14,8), dpi=100, sharex=True, sharey=True)
axes = axes.flatten()

for ax, df, label in zip(axes, fft_dfs, labels):

    # ---- Detect peaks (limit to 0–1000 Hz if you want)
    df_limited = df.loc[df["frequency"].between(0, 50000)]
    peaks = detect_fft_peaks(df_limited, neighborhood=100, top_n=10)

    # ---- Print peaks
    print(f"\n=============== Top Peaks ({label}) ===============")
    print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
    for i, row in enumerate(peaks.itertuples(), start=1):
        print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")
    print("====================================================")

    # ---- Plot FFT
    ax.plot(df["frequency"], df["magnitude"], lw=1.2)

    # ---- Plot peaks
    ax.scatter(peaks["frequency"], peaks["magnitude"], 
               color="red", zorder=5)

    # ---- Annotate peaks
    for _, row in peaks.iterrows():
        ax.text(row["frequency"],
                row["magnitude"] * 1.05,
                f'{row["frequency"]:.1f}',
                rotation=45,
                ha='center',
                fontsize=8)

    ax.set_title(label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

fig.suptitle("Fourier Profiles, full range", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show() if not save_outputs else None

if save_outputs:
    fig.savefig(output_dir / "fft_comparison_2x2_grid_with_peaks.png", dpi=300)
# %%
#%% -------- 2x2 GRID 0–1 kHz RANGE WITH PEAKS --------

fig, axes = plt.subplots(2, 2, figsize=(14,8), dpi=100, sharex=True, sharey=True)
axes = axes.flatten()

for ax, df, label in zip(axes, fft_dfs, labels):

    # ---- Limit to 0–1 kHz
    df_limited = df.loc[df["frequency"].between(0, 1000)]

    # ---- Detect top 10 peaks
    peaks = detect_fft_peaks(df_limited, neighborhood=1, top_n=10)

    # ---- Print peaks
    print(f"\n=============== Top Peaks ({label}, 0–1 kHz) ===============")
    print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
    for i, row in enumerate(peaks.itertuples(), start=1):
        print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")
    print("====================================================")

    # ---- Plot FFT limited to 0–1 kHz
    ax.plot(df_limited["frequency"], df_limited["magnitude"], lw=1.2)

    # ---- Plot peaks
    ax.scatter(peaks["frequency"], peaks["magnitude"], color="red", zorder=5)

    # ---- Annotate peaks
    for _, row in peaks.iterrows():
        ax.text(row["frequency"],
                row["magnitude"] * 1.05,
                f'{row["frequency"]:.1f}',
                rotation=45,
                ha='center',
                fontsize=8)

    ax.set_title(label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

fig.suptitle("Fourier Profiles, 0–1 kHz range", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show() if not save_outputs else None

if save_outputs:
    fig.savefig(output_dir / "fft_comparison_2x2_grid_0_1kHz_with_peaks.png", dpi=300)
# %%
