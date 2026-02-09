#%%
import pandas as pd
from pathlib import Path
import re

#==========================
# USER INPUT
#==========================
home_dir = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026")  # top-level directory containing all run subdirs
txt_list_file = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/files.txt")  # file listing all .txt filenames
excel_file = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/logbook_Z.xlsx")  # existing Excel file

#==========================
# LOAD EXCEL
#==========================
df_excel = pd.read_excel(excel_file)

#==========================
# LOAD RUN NAMES
#==========================
with open(txt_list_file, "r") as f:
    run_files = [line.strip() for line in f if line.strip()]

#==========================
# EXTRACT BASELINES
#==========================
baseline_dict = {}  # run_name -> baseline value

for run_file in run_files:
    run_name = run_file.replace(".txt", "")
    log_file = home_dir / run_name / "analysis.log"

    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                if "Baseline:" in line:
                    # Extract the numeric value after 'Baseline:'
                    match = re.search(r"Baseline:\s*([0-9.]+)", line)
                    if match:
                        baseline_value = float(match.group(1))
                        baseline_dict[run_file] = baseline_value
                    break  # stop after first match
    else:
        print(f"Warning: {log_file} does not exist")
        baseline_dict[run_file] = None

#==========================
# INSERT BASELINE INTO EXCEL
#==========================
df_excel["baseline"] = df_excel["filename"].map(baseline_dict)

#==========================
# SAVE UPDATED EXCEL
#==========================
output_excel = excel_file.parent / f"{excel_file.stem}_with_baselines.xlsx"
df_excel.to_excel(output_excel, index=False)

print(f"Updated Excel saved to {output_excel}")
