#%%
import os
import matplotlib.pyplot as plt
plt.style.use('/home/esamouil/Downloads/pub_clean.mplstyle')
#%%
base_path = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts"

indices = []
baselines = []
stds = []
fit_sigmas = []

for i in range(108):

    log_file = os.path.join(
        base_path,
        f"Z220126_IBM1_0008_part{i}",
        "analysis.log"
    )

    if not os.path.isfile(log_file):
        print(f"Missing: part{i}")
        continue

    baseline = None
    std = None
    fit_sigma = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Baseline (" in line and "μV" in line:
                baseline = float(line.split(":")[1].strip())

            if "Standard Deviation (" in line and "μV" in line:
                std = float(line.split(":")[1].strip())

            if "Std σ" in line and "=" in line:
                fit_sigma = float(line.split("=")[1].strip())

    if baseline is not None and std is not None and fit_sigma is not None:
        indices.append(i)
        baselines.append(baseline)
        stds.append(std)
        fit_sigmas.append(fit_sigma)
    else:
        print(f"Missing values in part{i}")

#%%
# ---- Print table ----
print("\nExtracted Values:\n")
print(f"{'Part':<8}{'Baseline (μV)':<20}{'STD (μV)':<20}{'Fit σ':<15}")
print("-"*63)

for i, b, s, fs in zip(indices, baselines, stds, fit_sigmas):
    print(f"{i:<8}{b:<20.4f}{s:<20.4f}{fs:<15.4f}")

#%%
#%%
import numpy as np

# Convert index to hours (0 → 18 hours)
hours = np.linspace(0, 18, len(indices))

FIG_W = 16
FIG_H = 4

# ---- Plot baseline ----
plt.figure(figsize=(FIG_W, FIG_H))
plt.plot(hours, baselines)
plt.xlabel("Time (hours)")
plt.ylabel("Baseline (μV)")
plt.title("Baseline vs Time")
plt.ylim( min(baselines)-100, max(baselines)+100 )  # adjust margin
plt.tight_layout()
plt.show()

# ---- Plot std (μV) ----
plt.figure(figsize=(FIG_W, FIG_H))
plt.plot(hours, stds)
plt.xlabel("Time (hours)")
plt.ylabel("Standard Deviation (μV)")
plt.title("STD (μV) vs Time")
plt.tight_layout()
plt.show()

# ---- Plot Gaussian fit sigma ----
plt.figure(figsize=(FIG_W, FIG_H))
plt.plot(hours, fit_sigmas)
plt.xlabel("Time (hours)")
plt.ylabel("Gaussian Fit σ")
plt.title("Gaussian Fit Sigma vs Time")
plt.tight_layout()
plt.show()
#%%
# ---- Histogram of baselines ----
# plt.figure(figsize=(8, 4))
# plt.hist(baselines, bins=6)
# plt.xlabel("Baseline (μV)")
# plt.ylabel("Counts")
# plt.title("Distribution of Baseline Values")
# plt.tight_layout()
# plt.show()
#%%