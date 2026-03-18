import json
import subprocess

files_list = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts/files.txt"
config_file = "config_large_run.json"
python_script = "analysis_script_large_run.py"

with open(files_list) as f:
    parquet_files = [line.strip() for line in f if line.strip()]

with open(config_file) as f:
    config = json.load(f)

total = len(parquet_files)
print(f"Found {total} parquet parts.\n")

for i, parquet_name in enumerate(parquet_files, 1):

    print(f"[{i}/{total}] {parquet_name}")

    config["parquet_file"] = parquet_name

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    subprocess.run(["python3", python_script])

print("\nAll parts processed.")