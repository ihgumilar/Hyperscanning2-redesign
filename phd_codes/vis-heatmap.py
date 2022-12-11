# %%
import sys
from typing import List

import matplotlib.cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from EEG.stats import Connections
from eye_tracker.analysis import EyeAnalysis

# add alpha (transparency) to a colormap (with background picture)
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

# %% Combine dataframe of eye tracker
eye_analysis = EyeAnalysis()

path2eyefiles = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/raw_cleaned_combined_experimental_eye_data/"

df_avert_pre_odd, df_avert_pre_even = eye_analysis.combine_eye_data(
    path2eyefiles, "averted_pre"
)

# %% Prepare the data so that ready to be plotted

# Make Copy of odd and even
df_avert_pre_odd_new = df_avert_pre_odd.copy(deep=True)
df_avert_pre_even_new = df_avert_pre_even.copy(deep=True)

# Change FoveaEven to Fovea and FoveaOdd to Fovea
df_avert_pre_odd_new.rename(columns={"FoveaOdd": "Fovea"}, inplace=True)
df_avert_pre_even_new.rename(columns={"FoveaEven": "Fovea"}, inplace=True)

# Combine dataframe odd and even
df_combined_averted_pre = pd.concat(
    [df_avert_pre_odd_new, df_avert_pre_even_new], ignore_index=True
)

# Round off columns of GazeDirectionRight column
df_combined_averted_pre = df_combined_averted_pre.round(
    {"GazeDirectionRight(X)": 1, "GazeDirectionRight(Y)": 1}
)

# Group multiple columns and reset the index
df_group = (
    df_combined_averted_pre.groupby(["GazeDirectionRight(X)", "GazeDirectionRight(Y)"])
    .size()
    .reset_index(name="count")
)
# Count how many data that are under the same coordinate or spot
pivot = df_group.pivot(
    index="GazeDirectionRight(X)", columns="GazeDirectionRight(Y)", values="count"
)

# Convert dataframe to numpy array
pivot_array = pivot.to_numpy()

# %% Plot the data with background picture

fig, ax = plt.subplots(figsize=(7, 7))
wd = matplotlib.cm.winter._segmentdata  # only has r,g,b
wd["alpha"] = ((0.0, 0.0, 0.3), (0.3, 0.3, 1.0), (1.0, 1.0, 1.0))

# modified colormap with changing alpha
al_winter = LinearSegmentedColormap("AlphaWinter", wd)


# get the map IMAGE as an array so we can plot it
map_img = mpimg.imread("averted_eye.jpg")

# making and plotting heatmap

colormap = sns.color_palette("Spectral", as_cmap=True)

sns.set()

# This is where we plot the data
hmax = sns.heatmap(
    pivot_array,
    # cmap = al_winter, # this worked but I didn't like it
    cmap=colormap,
    alpha=0.48,  # whole heatmap is translucent
    annot=False,
    zorder=2,
    cbar_kws={"label": "Gaze-Direction Intensity"},
)


# heatmap uses pcolormesh instead of imshow, so we can't pass through
# extent as a kwarg, so we can't mmatch the heatmap to the map. Instead,
# match the map to the heatmap:

# This is where we put the background picture
hmax.imshow(
    map_img,
    aspect=hmax.get_aspect(),
    extent=hmax.get_xlim() + hmax.get_ylim(),
    zorder=1,
)  # put the map under the heatmap

ax.set_title("Averted-Eye Gaze Condition")

ax.tick_params(axis="both", which="both", length=0)
# Remove x labels
ax.set(xticklabels=[], yticklabels=[])

# Color bar related thing
c_bar = hmax.collections[0].colorbar
c_bar.set_ticks([10000, 80000])
c_bar.set_ticklabels(["low", "High"])

# %% Save the figures
fig = hmax.get_figure()
fig.savefig("figures/averted_pre150.png", dpi=100)
