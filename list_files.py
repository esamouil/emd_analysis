from pathlib import Path

# === CONFIG ===
folder_path = Path("/home/esamouil/data_ess__/Local_CC_2025/001_odin_bms_cc_2025/ibm2")  # change this
output_file = folder_path / "files.txt"

# find all .txt and .emd files
txt_files = list(folder_path.glob("*.txt"))
emd_files = list(folder_path.glob("*.emd"))

# get stems of .emd files
emd_stems = {f.stem for f in emd_files}

# only keep txt files that have a matching emd stem
matched_txt_files = [f.name for f in txt_files if f.stem in emd_stems]

# save to files.txt
with open(output_file, "w") as f:
    for txt in matched_txt_files:
        f.write(txt + "\n")

print(f"Saved {len(matched_txt_files)} matched .txt files to {output_file}")
