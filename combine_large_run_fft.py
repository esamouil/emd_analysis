#%%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from esamouil_functions import *

#%% Configuring plot style
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')

#%%
# User inputs
base_folder = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts")
output_file = base_folder / "combined_fft.parquet"

# Find all fft_dataframe.parquet files
fft_files = sorted(base_folder.glob("Z220126_IBM1_0008_part*/fft/fft_dataframe.parquet"))

print("Found FFT files:")
for f in fft_files:
    print(f)

if not fft_files:
    raise ValueError("No FFT parquet files found!")

#%% Combine FFTs
combined_df = pd.read_parquet(fft_files[0])
combined_df["magnitude"] = combined_df["magnitude"] ** 2  # square magnitudes

for fft_file in fft_files[1:]:
    df = pd.read_parquet(fft_file)
    combined_df["magnitude"] += df["magnitude"] ** 2
    del df

combined_df["magnitude"] = np.sqrt(combined_df["magnitude"])

# Save combined parquet
combined_df.to_parquet(output_file, index=False)
print(f"Combined FFT saved to {output_file}")

#%%
# ---------- Full FFT plot ----------
fig, ax = plt.subplots(figsize=(16,4))
ax.plot(combined_df["frequency"], combined_df["magnitude"], lw=0.7)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Combined FFT - Full Range")
ax.grid(True)
plt.tight_layout()
plt.show()

# Detect top 10 peaks
peaks_full = detect_fft_peaks(combined_df, neighborhood=10, top_n=10)
fig, ax = plot_with_peaks(combined_df, peaks_full)
ax.set_title("Combined FFT with Top 10 Peaks - Full Range")
plt.show()

# Print peaks
print("=============== Top 10 Peaks (Full FFT) ===============")
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
for i, row in enumerate(peaks_full.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")
print("=======================================================\n")

#%%
# ---------- FFT 0-1kHz ----------
fft_0_1k = combined_df.loc[combined_df["frequency"].between(0, 1000)]

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(fft_0_1k["frequency"], fft_0_1k["magnitude"], lw=0.7)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Combined FFT - 0-1 kHz Range")
ax.grid(True)
plt.tight_layout()
plt.show()

# Detect top 10 peaks in 0-1kHz range
peaks_0_1k = detect_fft_peaks(fft_0_1k, neighborhood=1, top_n=10)
fig, ax = plot_with_peaks(fft_0_1k, peaks_0_1k)
ax.set_title("Combined FFT with Top 10 Peaks - 0-1 kHz")
plt.show()

# Print peaks
print("=============== Top 10 Peaks (0-1 kHz) ===============")
print(f"{'No.':>4} {'frequency':>12} {'magnitude':>12} {'FWHM':>12}")
for i, row in enumerate(peaks_0_1k.itertuples(), start=1):
    print(f"{i:>4} {row.frequency:12.2f} {row.magnitude:12.2g} {row.FWHM:12.2g}")
print("=======================================================\n")
# %%
#%%
# ---------- Noise Metrics ----------
metrics_df = calculate_noise_metrics_from_single_df(combined_df)

print("=============== Noise Metrics ===============") 
print("Low: 0-1 kHz\tMid: 1-10 kHz\tHigh: >10 kHz\n")

for col in metrics_df.columns:
    print(f"{col}: {metrics_df[col][0]:.2f}")

# Compute Nyquist frequency and max frequency resolution
f_nyquist = combined_df["frequency"].max()
delta_f = combined_df["frequency"].diff().min()
print(f"Nyquist frequency (Hz): {f_nyquist:.2f}")
print(f"Max Frequency resolution (Hz): {delta_f:.2g}")
print("=============================================\n")
# %%
# ----------
# Combine Histograms Conservatively
# Each part has 'voltage/histogram_dataframe.parquet'
hist_files = sorted(base_folder.glob("Z220126_IBM1_0008_part*/voltage/histogram_dataframe.parquet"))

print("Found Histogram files:")
for f in hist_files[:2]:  # print first 2 contents
    df_print = pd.read_parquet(f)
    print(df_print.head(5))
    print("...")  # just to indicate more rows

dfs = [pd.read_parquet(f) for f in hist_files if not pd.read_parquet(f).empty]

if not dfs:
    raise ValueError("No non-empty histograms found!")

# Determine smallest bin width and range across all histograms
bin_widths = [np.diff(df["bin_center"]).min() for df in dfs if len(df) > 1]
bin_width = min(bin_widths)
bin_min = min(df["bin_center"].min() for df in dfs)
bin_max = max(df["bin_center"].max() for df in dfs)

# Define common bins edges (will use them for summing counts exactly)
common_edges = np.arange(bin_min - bin_width/2, bin_max + 1.5*bin_width, bin_width)
common_centers = (common_edges[:-1] + common_edges[1:]) / 2

combined_counts = np.zeros_like(common_centers)

# Add counts from each histogram into nearest bin
for df in dfs:
    for c, count in zip(df["bin_center"], df["counts"]):
        # find nearest bin index
        idx = np.argmin(np.abs(common_centers - c))
        combined_counts[idx] += count

# Create final combined histogram dataframe
combined_hist_df = pd.DataFrame({
    "bin_center": common_centers,
    "counts": combined_counts
})

print("Total counts in combined histogram:", combined_hist_df["counts"].sum())

# %%
# ---------- Plot Combined Histogram ----------
fig, ax = plt.subplots(figsize=(16,4))
ax.bar(
    combined_hist_df["bin_center"],
    combined_hist_df["counts"],
    width=np.diff(combined_hist_df["bin_center"]).mean(),
    alpha=0.7
)
ax.set_xlabel("Bin Center")
ax.set_ylabel("Counts")
ax.set_title("Combined Histogram")
ax.grid(True)
plt.tight_layout()
plt.show()

print(combined_hist_df.head())
print(combined_hist_df.tail())
print(combined_hist_df.to_string())
# %%
# %%
# ---------- Combined Histogram Gaussian Fit & Plot (with range) ----------

from scipy.optimize import curve_fit

# User-defined range
bin_min = -1000  # adjust as needed
bin_max = 1000   # adjust as needed

# Filter combined histogram within range
mask = (combined_hist_df["bin_center"] >= bin_min) & (combined_hist_df["bin_center"] <= bin_max)
subset_df = combined_hist_df.loc[mask]

# Skip if empty
if subset_df.empty:
    raise ValueError("No bins in the specified range for combined histogram.")

bin_centers = subset_df["bin_center"].to_numpy()
counts = subset_df["counts"].to_numpy()

# Gaussian function
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Initial guess: amplitude, mean, sigma
p0 = [counts.max(), np.average(bin_centers, weights=counts), np.sqrt(np.average((bin_centers - np.average(bin_centers, weights=counts))**2, weights=counts))]
popt, _ = curve_fit(gauss, bin_centers, counts, p0=p0)
A, mu, sigma = popt

# Plot
fig, ax = plt.subplots(figsize=(10,4))
ax.bar(bin_centers, counts, width=np.diff(bin_centers).mean(), alpha=0.6, edgecolor='k', label='Combined Histogram')
ax.plot(bin_centers, gauss(bin_centers, A, mu, sigma), 'r-', lw=2, label='Gaussian fit')

# Stats box
textstr = f"μ = {mu:.3e}\nσ = {sigma:.3e}\nA = {A:.1f}"
ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

ax.set_title("Signal Values Histogram")
ax.set_xlabel("Signal (μV)")
ax.set_ylabel("Counts")
ax.legend()
fig.tight_layout()
plt.show()

# Chi² / NDF
errors = np.sqrt(counts)
errors[errors == 0] = 1.0
chi2 = np.sum(((counts - gauss(bin_centers, *popt)) / errors)**2)
ndf = len(counts) - len(popt)
chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

print("\n===== Gaussian Fit Statistics (Subset) =====")
print(f"Amplitude A = {A:.2f}")
print(f"Mean μ      = {mu:.3e}")
print(f"Std σ       = {sigma:.3e}")
print(f"Chi²        = {chi2:.2f}")
print(f"NDF         = {ndf}")
print(f"Chi² / NDF  = {chi2_ndf:.2f}")
print("============================================\n")# %%

# %%
