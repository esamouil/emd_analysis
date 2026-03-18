#%%
import pandas as pd
import matplotlib.pyplot as plt

# Set the path to your TOF file
file_path = '/home/esamouil/analysis/data_stor/psi_aug_2024_min_data_set/11ibm_meas_20240803_cdre12_as20_1001.tof'  # <-- change this to your actual file

# Read the TOF file (whitespace-separated, no header)
df = pd.read_csv(file_path, header=None, delimiter='\s+')
df.columns = ['Timestamp', 'ADC']

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['ADC'], marker='.', linestyle='-')
plt.title(f'TOF Data: {file_path}')
plt.xlabel('Timestamp')
plt.ylabel('ADC Value')
plt.grid(True)
plt.show()

# %%
