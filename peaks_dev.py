# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# %% Load FFT data
fft_df = pd.read_csv("fft_data.csv")  # must have 'frequency' and 'magnitude'
freqs = fft_df["frequency"].to_numpy()
mags  = fft_df["magnitude"].to_numpy()

# %% Parameters
neighborhood = 1000  # number of points before and after
top_n = 200           # keep only the largest peaks

# %% Local maxima detection
peaks = []

for i in range(neighborhood, len(mags) - neighborhood):
    window_prev = mags[i - neighborhood:i]
    window_next = mags[i + 1:i + 1 + neighborhood]
    
    if mags[i] > np.max(window_prev) and mags[i] > np.max(window_next):
        peaks.append(i)

peaks = np.array(peaks)

# %% Keep only the top N peaks by magnitude
if len(peaks) > top_n:
    top_indices = np.argsort(mags[peaks])[::-1][:top_n]
    peaks = peaks[top_indices]

# %% Calculate FWHM for each detected peak
fwhm_list = []

for idx in peaks:
    m_peak = mags[idx]
    half = m_peak / 2
    
    # search left
    left_idx = idx
    while left_idx > 0 and mags[left_idx] > half:
        left_idx -= 1
    
    # search right
    right_idx = idx
    while right_idx < len(mags) - 1 and mags[right_idx] > half:
        right_idx += 1
    
    # FWHM in frequency
    fwhm = freqs[right_idx] - freqs[left_idx]
    fwhm_list.append(fwhm)

# %% Build dataframe of peaks
df_peaks = pd.DataFrame({
    "frequency": freqs[peaks],
    "magnitude": mags[peaks],
    "FWHM": fwhm_list
}).sort_values("magnitude", ascending=False).reset_index(drop=True)

print("Top detected peaks with FWHM:")
print(df_peaks)

# %% Interactive plot
fig = go.Figure()

# FFT line
fig.add_trace(go.Scatter(
    x=freqs,
    y=mags,
    mode='lines',
    name='FFT magnitude',
    line=dict(color='blue', width=1)
))

# Peaks
fig.add_trace(go.Scatter(
    x=freqs[peaks],
    y=mags[peaks],
    mode='markers',
    name=f'Top {top_n} peaks',
    marker=dict(color='red', size=8, symbol='x')
))

# Optional: show FWHM as horizontal lines
for idx, fwhm in zip(peaks, fwhm_list):
    half_mag = mags[idx] / 2
    left = freqs[idx] - fwhm/2
    right = freqs[idx] + fwhm/2
    fig.add_trace(go.Scatter(
        x=[left, right],
        y=[half_mag, half_mag],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        showlegend=False
    ))

fig.update_layout(
    title='FFT local-maxima peak detection with FWHM',
    xaxis_title='Frequency',
    yaxis_title='Magnitude',
    hovermode='closest',
    width=1000,
    height=500
)

fig.show()

# %%


# %%
# import numpy as np
# import pandas as pd

# def detect_fft_peaks(df_fft, neighborhood=1000, top_n=10):
#     """
#     Detect peaks in an FFT dataframe using neighborhood local-maxima method
#     and calculate their FWHM.

#     Parameters
#     ----------
#     df_fft : pd.DataFrame
#         Must have columns ['frequency', 'magnitude'].
#     neighborhood : int
#         Number of points before and after each point to compare.
#     top_n : int
#         Number of largest peaks to keep.

#     Returns
#     -------
#     df_peaks : pd.DataFrame
#         Columns: ['frequency', 'magnitude', 'FWHM'] for top peaks.
#     """

#     freqs = df_fft["frequency"].to_numpy()
#     mags = df_fft["magnitude"].to_numpy()

#     # -- local maxima detection --
#     peaks = []
#     for i in range(neighborhood, len(mags) - neighborhood):
#         if mags[i] > np.max(mags[i - neighborhood:i]) and mags[i] > np.max(mags[i + 1:i + 1 + neighborhood]):
#             peaks.append(i)
#     peaks = np.array(peaks)

#     # -- keep only top N peaks --
#     if len(peaks) > top_n:
#         top_indices = np.argsort(mags[peaks])[::-1][:top_n]
#         peaks = peaks[top_indices]

#     # -- calculate FWHM --
#     fwhm_list = []
#     for idx in peaks:
#         m_peak = mags[idx]
#         half = m_peak / 2
#         # search left
#         left_idx = idx
#         while left_idx > 0 and mags[left_idx] > half:
#             left_idx -= 1
#         # search right
#         right_idx = idx
#         while right_idx < len(mags) - 1 and mags[right_idx] > half:
#             right_idx += 1
#         fwhm = freqs[right_idx] - freqs[left_idx]
#         fwhm_list.append(fwhm)

#     # -- build dataframe --
#     df_peaks = pd.DataFrame({
#         "frequency": freqs[peaks],
#         "magnitude": mags[peaks],
#         "FWHM": fwhm_list
#     }).sort_values("magnitude", ascending=False).reset_index(drop=True)

#     return df_peaks

# %%
