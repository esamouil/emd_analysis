import json
import subprocess
from pathlib import Path

# Paths
files_list = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/files.txt"
config_file = "config.json"
python_script = "analysis_script.py"

# Read all txt filenames
with open(files_list, "r") as f:
    txt_files = [line.strip() for line in f if line.strip()]

total_files = len(txt_files)
print(f"Found {total_files} files to process.\n")

# Loop with simple run number
for i, txt_file in enumerate(txt_files, start=1):
    txt_path = Path(txt_file)
    print(f"Processing file {i}/{total_files}: {txt_path.name}")

    # Update config.json for current file
    with open(config_file, "r") as f:
        config = json.load(f)

    config["single_file_name"] = txt_path.name

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    # Run the main analysis script
    subprocess.run(["python3", python_script])

    print(f"Finished file {i}/{total_files}: {txt_path.name}")
    print("----------------------------")

print("\nAll files processed.")
