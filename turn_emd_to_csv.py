from esamouil_functions import *

# ===== USER SETTINGS =====
EMD_FILE = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0022.emd"
OUTPUT_CSV = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z210126_IBM0_0022.csv"
# =========================


df = process_binary_to_csv_chunked(EMD_FILE, verbose=True)
df.to_csv(OUTPUT_CSV, index=False)
