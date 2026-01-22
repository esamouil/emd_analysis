#%%
import numpy as np
import pandas as pd
from esamouil_functions import *
import json 
import matplotlib.pyplot as plt
from pathlib import Path


# To run this properly you either have to edit the fft_df function to take timestamps i
# seconds or to give it the timestamps in microseconds.

#%%
# -----------------------------
# user-defined parameters
# -----------------------------
A1 = 1.0       # amplitude of first sine
f1 = 1        # frequency of first sine (Hz)
A2 = 2
f2 = 4.5


duration = 10     # total time (s)
dt = 1e-4          # time between samples (s)
# -----------------------------

# time axis
t = np.arange(0, duration, dt)

# signal
signal = A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t)

# dataframe
df = pd.DataFrame({
    "timestamp": t,
    "adc_value": signal
})

print(df.head())

# %%
plot_df_range(df, x_col="timestamp", y_col="adc_value", x_min=0, x_max=10,
              title="signal", xlabel="time", ylabel="signal")

# %%
fft_df = fft_dataframe(df)
print(fft_df.head())

# %%
plot_df_range(fft_df, x_col="frequency", y_col="magnitude", x_min=0, x_max=10,
              title="FFT", xlabel="Hz", ylabel="Amplitude")

# %%
