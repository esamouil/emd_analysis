#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
#%%
# Path to saved FFT dataframe
fft_file = Path("/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM1_0008_parts/Z220126_IBM1_0008_part0/fft/fft_dataframe.parquet")  # change this

# Load dataframe
fft_df = pd.read_parquet(fft_file)
#%%
# Basic plot
plt.figure(figsize=(12, 5))
plt.plot(fft_df["frequency"], fft_df["magnitude"], lw=1)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of Run")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
print(fft_df.head())
print(fft_df.tail())
# %%
