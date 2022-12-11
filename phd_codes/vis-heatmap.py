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

# %% Make Copy of odd and even
df_avert_pre_odd_new = df_avert_pre_odd.copy(deep=True)
df_avert_pre_even_new = df_avert_pre_even.copy(deep=True)

# Change FoveaEven to Fovea and FoveaOdd to Fovea
df_avert_pre_odd_new.rename(columns={"FoveaOdd": "Fovea"}, inplace=True)
df_avert_pre_even_new.rename(columns={"FoveaEven": "Fovea"}, inplace=True)

# Combine dataframe odd and even
df_combined_averted_pre = pd.concat(
    [df_avert_pre_odd_new, df_avert_pre_even_new], ignore_index=True
)

# %% Get maximum value from the following columns. We use this info to create grid
print(
    df_combined_averted_pre[
        [
            "GazeDirectionRight(X)",
            "GazeDirectionRight(Y)",
            "GazeDirectionLeft(X)",
            "GazeDirectionLeft(Y)",
        ]
    ].max()
)

# %% Get minimum value from the following columns. We use this info to create grid
print(
    df_combined_averted_pre[
        [
            "GazeDirectionRight(X)",
            "GazeDirectionRight(Y)",
            "GazeDirectionLeft(X)",
            "GazeDirectionLeft(Y)",
        ]
    ].min()
)

# Plot GazeDirectionRight (X & Y)
df_combined_averted_pre.plot()

# %%
# If you want to customize the round off by individual columns
df_combined_averted_pre = df_combined_averted_pre.round(
    {"GazeDirectionRight(X)": 1, "GazeDirectionRight(Y)": 1}
)


# %% Plot with background picture
# Group multiple columns and reset the index
df_group = (
    df_combined_averted_pre.groupby(["GazeDirectionRight(X)", "GazeDirectionRight(Y)"])
    .size()
    .reset_index(name="count")
)

pivot = df_group.pivot(
    index="GazeDirectionRight(X)", columns="GazeDirectionRight(Y)", values="count"
)

# Convert dataframe to numpy array
pivot_array = pivot.to_numpy()

# %% Plot with background picture


fig, ax = plt.subplots(figsize=(8.5, 8.5))
wd = matplotlib.cm.winter._segmentdata  # only has r,g,b
wd["alpha"] = ((0.0, 0.0, 0.3), (0.3, 0.3, 1.0), (1.0, 1.0, 1.0))

# modified colormap with changing alpha
al_winter = LinearSegmentedColormap("AlphaWinter", wd)


# get the map IMAGE as an array so we can plot it
map_img = mpimg.imread("averted_eye.jpg")

# making and plotting heatmap
# colormap = matplotlib.cm.winter
# colormap = sns.color_palette("magma", as_cmap=True)
colormap = sns.color_palette("Spectral", as_cmap=True)

sns.set()


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

hmax.imshow(
    map_img,
    aspect=hmax.get_aspect(),
    extent=hmax.get_xlim() + hmax.get_ylim(),
    zorder=1,
)  # put the map under the heatmap

ax.set_title("Averted-Eye Gaze Condition")

# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
# # Remove ticks for both x and y axis
ax.tick_params(axis="both", which="both", length=0)
# # Remove x labels
ax.set(xticklabels=[], yticklabels=[])

# # Get the legend labels of colorbar
# legend_labels = [x.get_text() for x in cbar_ax.get_yticklabels()]
# legend_labels_new = []

# # Change old legend label with new labels
# for idx, val in enumerate(legend_labels):

#     if idx == 1:
#         val = "Low"
#         legend_labels_new.append(val)

#     elif idx == (len(legend_labels) - 2):
#         val = "High"
#         legend_labels_new.append(val)
#     else:
#         val = ""
#         legend_labels_new.append(val)

# # set the yticklabels to the new labels we just created.
# cbar_ax.set_yticklabels(legend_labels_new)

# Remove x and y ticks in color bar
# ax.tick_params(axis="both", which="both", length=0)
# Specify labels for specific ticks in the colorbar
# c_bar = hmax.colorbar(hmax, ticks=range(5))
# labels = ('null', 'exit', 'power', 'smile', 'null')
# c_bar.ax.set_yticklabels(labels)
# c_bar.set_ticks([-8, 0, 8 ])
# c_bar.set_ticklabels(["Low", "Medium", "High"])

plt.show()
# %%  Plot heatmap without background picture

# Group multiple columns and reset the index
df_group = (
    df_combined_averted_pre.groupby(["GazeDirectionRight(X)", "GazeDirectionRight(Y)"])
    .size()
    .reset_index(name="count")
)

pivot = df_group.pivot(
    index="GazeDirectionRight(X)", columns="GazeDirectionRight(Y)", values="count"
)
ax = sns.heatmap(pivot, cbar_kws={"label": "Looking duration"})
plt.show()


# %% Testing changing legend of colorbar, remove x & y axis heatmap

import seaborn as sns

sns.set()
import numpy as np

np.random.seed(0)
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
fig.set_size_inches(14, 7)
uniform_data = np.random.rand(10, 12)
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])

# Plot the data
sns.heatmap(
    uniform_data, ax=ax, cbar_ax=cbar_ax, cbar_kws={"label": "Intensity of looking"}
)

ax = sns.heatmap(
    uniform_data, ax=ax, cbar_ax=cbar_ax, cbar_kws={"label": "Intensity of looking"}
)
# Remove ticks for both x and y axis
ax.tick_params(axis="both", which="both", length=0)
# Remove x labels
ax.set(xticklabels=[], yticklabels=[])

# Get the legend labels of colorbar
legend_labels = [x.get_text() for x in cbar_ax.get_yticklabels()]
legend_labels_new = []

# Change old legend label with new labels
for idx, val in enumerate(legend_labels):

    if idx == 1:
        val = "Low"
        legend_labels_new.append(val)

    elif idx == (len(legend_labels) - 2):
        val = "High"
        legend_labels_new.append(val)
    else:
        val = ""
        legend_labels_new.append(val)

# set the yticklabels to the new labels we just created.
cbar_ax.set_yticklabels(legend_labels_new)
# Remove x and y ticks in color bar
cbar_ax.tick_params(axis="both", which="both", length=0)

# ToDO : Change color so that it can be transparent. See if it is already available in the available code above

# %%
