#%%
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

#%%
# === Plot Style Config ===
plt.rcParams.update({
    # Font
    "font.family": "Nimbus Roman",
    "mathtext.rm": "Nimbus Roman",
    "font.size": 14,

    # Figure
    "figure.figsize": (6, 4),
    "figure.dpi": 100,

    # Lines and markers
    "lines.linewidth": 2,
    "lines.markersize": 6,

    # Axes labels and ticks
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Grid
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.alpha": 0.7,

    # Legend
    "legend.fontsize": 12,

    # Error bars
    "errorbar.capsize": 4,

    # Boxplots
    "boxplot.flierprops.markersize": 4,
    "boxplot.meanprops.markersize": 4,


    #colour palette
    "axes.prop_cycle": plt.cycler(color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
)
})

#%%
# === Test Plot with multiple lines ===
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2*x)
y4 = np.exp(-0.1*x) * np.sin(3*x)

plt.plot(x, y1, label="sin(x)")
plt.plot(x, y2, label="cos(x)")
plt.plot(x, y3, label="sin(2x)")
plt.plot(x, y4, label="exp(-0.1x)*sin(3x)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Style Test Plot with Multiple Lines")
plt.legend()
plt.tight_layout()
plt.show()

# %%
