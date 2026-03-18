#%%
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
#%%
# ======================
# USER SETTINGS
# ======================
input_file = "/home/esamouil/analysis/data_stor/local_commissioning_data/004_dream_cc_2026/Z220126_IBM0_0003.osc"   # your file
dt_us = 10.0               # time between samples (microseconds)
output_wav = "signal.wav"
#%%
# ======================
# READ FILE
# ======================
data = np.loadtxt(input_file)
#%%
adc = data[:, 1]
#%%
# ======================
# TIME / SAMPLING
# ======================
dt = dt_us * 1e-6          # us → seconds
fs = int(1 / dt)           # sampling rate
#%%
print(f"Sampling rate: {fs} Hz")
#%%
# ======================
# BASELINE + SHIFT
# ======================
baseline = np.mean(adc)
adc_shifted = adc - baseline

print(f"Baseline: {baseline:.2f}")
#%%
# ======================
# NORMALIZE
# ======================
adc_norm = adc_shifted / np.max(np.abs(adc_shifted))
#%%
# ======================
# WRITE WAV
# ======================
audio_int16 = np.int16(adc_norm * 32767)
write(output_wav, fs, audio_int16)
#%%
print(f"WAV written to: {output_wav}")
#%%
# ======================
# PLAY AUDIO
# ======================
print("Playing audio...")
sd.play(adc_norm, fs)
sd.wait()

# %%
