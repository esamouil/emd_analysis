#%%
import os
#%%
base_path = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts"

for i in range(108):  # 0 to 107
    part_folder = f"Z220126_IBM1_0008_part{i}"
    sparks_path = os.path.join(base_path, part_folder, "voltage", "sparks")

    if not os.path.isdir(sparks_path):
        print(f"[MISSING] {sparks_path}")
        continue

    contents = os.listdir(sparks_path)

    if contents:
        print(f"[NOT EMPTY] part{i} -> {len(contents)} item(s)")
    else:
        print(f"[EMPTY] part{i}")
# %%
