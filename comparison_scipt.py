#%%
import pandas as pd
from pathlib import Path
from esamouil_functions import *
import matplotlib.pyplot as plt
import os
import json

#%% Files for comparison
run_files = {
    "Run1": {
        "txt": Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0004.txt"),
        "emd": Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0004.emd")
    },
    "Run2": {
        "txt": Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM0_0004.txt"),
        "emd": Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM0_0004.emd")
    }
}

save_outputs = False
output_dir = "./comparison_plots"
os.makedirs(output_dir, exist_ok=True)

#%% Excel metadata for cor_boolean
excel_path = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/logbook_Z_with_baselines.xlsx"
excel_df = pd.read_excel(excel_path)
file_info = {
    row['filename']: (row['act.det'], bool(row['corr. needed']))
    for _, row in excel_df.iterrows()
}

#%% Store all processed data for plotting
plot_data = {}

for run_name, paths in run_files.items():
    print(f"Processing {run_name}...\n")

    # Parse TXT
    info = parse_txt_file(paths["txt"])

    # Get cor_boolean from Excel
    current_file = paths["txt"].name
    if current_file in file_info:
        actual_detector, cor_boolean = file_info[current_file]
    else:
        cor_boolean = False

    # Load EMD data
    if info["event_binary"]:
        df = process_binary_to_parquet_3(paths["emd"], count_to=None, verbose=False)
    else:
        df = read_ascii_emd(paths["emd"])

    # Filter timestamps
    df = filter_df(df, 0,1000000)

    # Pre-TIA baseline & std
    baseline_pre = df["adc_value"].mean()
    std_pre = df["adc_value"].std()

    # Scale ADC values
    scale_factor = info["tia_summation_points"]
    if cor_boolean:
        scale_factor *= 4
    df["adc_value"] /= scale_factor

    # Baseline shift
    baseline_shifted, std_shifted = baseline_shift(df, [(0.0,1.0)])
    df["adc_value"] -= baseline_shifted

    # FFT
    fft_df = fft_dataframe(df)

    # Rebin FFT
    fft_df_rebinned = rebin_by_width(fft_df, 50)

    # Peaks
    peaks = find_top_peaks(fft_df, n=10)
    peaks_rebinned = find_top_peaks(fft_df_rebinned, n=10)

    # Noise metrics
    metrics_df = calculate_noise_metrics_from_single_df(fft_df)

    # Save everything for plotting
    plot_data[run_name] = {
        "df": df,
        "baseline_pre": baseline_pre,
        "std_pre": std_pre,
        "baseline_shifted": baseline_shifted,
        "std_shifted": std_shifted,
        "fft": fft_df,
        "fft_rebinned": fft_df_rebinned,
        "peaks": peaks,
        "peaks_rebinned": peaks_rebinned,
        "metrics": metrics_df
    }

#%% Plot raw ADC together
fig, ax = plt.subplots(figsize=(16,4))
for run_name, data in plot_data.items():
    ax.plot(data["df"]["timestamp"], data["df"]["adc_value"], label=run_name)
ax.set_xlabel("Timestamp (µs)")
ax.set_ylabel("ADC value")
ax.set_title("Raw ADC Comparison")
ax.legend()
plt.tight_layout()
plt.show()
if save_outputs: fig.savefig(os.path.join(output_dir, "raw_adc_comparison.png"))

#%% Plot last 2% of ADC together
fig, ax = plt.subplots(figsize=(16,4))
fraction = 0.02
for run_name, data in plot_data.items():
    df_tail = data["df"].iloc[int((1-fraction)*len(data["df"])):]
    ax.plot(range(len(df_tail)), df_tail["adc_value"], label=run_name)
ax.set_xlabel("Line number (last 2%)")
ax.set_ylabel("ADC value")
ax.set_title("Last 2% ADC Comparison")
ax.legend()
plt.tight_layout()
plt.show()
if save_outputs: fig.savefig(os.path.join(output_dir, "last2percent_comparison.png"))

#%% Plot FFT magnitude together
fig, ax = plt.subplots(figsize=(16,4))
for run_name, data in plot_data.items():
    ax.plot(data["fft"]["frequency"], data["fft"]["magnitude"], label=run_name)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("FFT Magnitude Comparison")
ax.legend()
plt.tight_layout()
plt.show()
if save_outputs: fig.savefig(os.path.join(output_dir, "fft_comparison.png"))

#%% Plot FFT peaks together
fig, ax = plt.subplots(figsize=(16,4))
for run_name, data in plot_data.items():
    ax.plot(data["fft"]["frequency"], data["fft"]["magnitude"], lw=0.5, label=run_name)
    ax.scatter(data["peaks"]["frequency"], data["peaks"]["magnitude"], marker="x")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("FFT Peaks Comparison")
ax.legend()
plt.tight_layout()
plt.show()
if save_outputs: fig.savefig(os.path.join(output_dir, "fft_peaks_comparison.png"))

#%% Plot rebinned FFT together
fig, ax = plt.subplots(figsize=(16,4))
for run_name, data in plot_data.items():
    ax.plot(data["fft_rebinned"]["frequency"], data["fft_rebinned"]["magnitude"], label=run_name)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Rebinned FFT Comparison")
ax.legend()
plt.tight_layout()
plt.show()
if save_outputs: fig.savefig(os.path.join(output_dir, "fft_rebinned_comparison.png"))

#%% Plot rebinned FFT peaks together
fig, ax = plt.subplots(figsize=(16,4))
for run_name, data in plot_data.items():
    ax.plot(data["fft_rebinned"]["frequency"], data["fft_rebinned"]["magnitude"], lw=0.5, label=run_name)
    ax.scatter(data["peaks_rebinned"]["frequency"], data["peaks_rebinned"]["magnitude"], marker="x")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Rebinned FFT Peaks Comparison")
ax.legend()
plt.tight_layout()
plt.show()
if save_outputs: fig.savefig(os.path.join(output_dir, "fft_rebinned_peaks_comparison.png"))

#%% Print noise metrics side by side
for run_name, data in plot_data.items():
    print(f"=== Noise metrics for {run_name} ===")
    print(data["metrics"])
    print("\n")

# %%
