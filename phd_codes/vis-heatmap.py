# %%
import os
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
from pandas import DataFrame
from scipy.stats import pearsonr


# %% Combine dataframe of eye tracker
def plot_heatmap(
    df_pre_odd: DataFrame,
    df_pre_even: DataFrame,
    df_post_odd: DataFrame,
    df_post_even: DataFrame,
    tag: str,
    save_fig=False,
    path2savefigure: str = os.getcwd(),
):

    """
    Objective  : Plot heatmap of gaze direction for a particular eye condition, i.e., averted, direct, and natural

    Parameters :
                 - df_pre_odd (DataFrame): Combined dataframe of pre training of odd subjects
                 - df_pre_even (DataFrame): Combined dataframe of pre training of even subjects
                 - df_post_odd (DataFrame): Combined dataframe of post training of odd subjects
                 - df_post_even (DataFrame): Combined dataframe of post training of even subjects
                                      NOTE : the above combined dataframe is the output of combine_eye_data function

                 - tag (str) : Choose one of the options below. MUST be the exact word below :
                                "averted", "direct", "natural"
                 - save_fig (boolean)=False (opt) : Option to save the figure. Default is not saving
                 - path2savefigure (str)=os.getcwd() (opt) : Set path to save the figure. Default is current working directory
    """

    odd_tag = tag.lower() + "_pre"
    even_tag = tag.lower() + "_post"

    # Prepare the data so that ready to be plotted

    # Change FoveaEven to Fovea and FoveaOdd to Fovea (Pre-training)
    df_pre_odd.rename(columns={"FoveaOdd": "Fovea"}, inplace=True)
    df_pre_even.rename(columns={"FoveaEven": "Fovea"}, inplace=True)

    # Change FoveaEven to Fovea and FoveaOdd to Fovea (Post-training)
    df_post_odd.rename(columns={"FoveaOdd": "Fovea"}, inplace=True)
    df_post_even.rename(columns={"FoveaEven": "Fovea"}, inplace=True)

    # Combine dataframe odd and even (Pre-training)
    df_combined_averted_pre = pd.concat([df_pre_odd, df_pre_even], ignore_index=True)

    # Combine dataframe odd and even (Post-training)
    df_combined_averted_post = pd.concat([df_post_odd, df_post_even], ignore_index=True)

    # Combine dataframe of Pre and Post-training
    df_combined_averted_pre_post = pd.concat(
        [df_combined_averted_pre, df_combined_averted_post], ignore_index=True
    )

    # Round off columns of GazeDirectionRight column
    df_combined_averted_pre_post = df_combined_averted_pre_post.round(
        {
            "GazeDirectionRight(X)": 1,
            "GazeDirectionRight(Y)": 1,
            "GazeDirectionLeft(X)": 1,
            "GazeDirectionLeft(Y)": 1,
        }
    )

    # Take only Gaze direction for both right and left eye
    df_combined_pre_post_right = df_combined_averted_pre_post[
        ["GazeDirectionRight(X)", "GazeDirectionRight(Y)"]
    ]
    df_combined_pre_post_left = df_combined_averted_pre_post[
        ["GazeDirectionLeft(X)", "GazeDirectionLeft(Y)"]
    ]

    # Rename the columns of left eye to be similar to right eye so that we can combine
    df_combined_pre_post_left.rename(
        columns={"GazeDirectionLeft(X)": "GazeDirectionRight(X)"}, inplace=True
    )
    df_combined_pre_post_left.rename(
        columns={"GazeDirectionLeft(Y)": "GazeDirectionRight(Y)"}, inplace=True
    )

    # Combine both right and left eye gaze directions into one dataframe (stacking each other - vertically)
    df_combined_averted_pre_post_gaze_only = pd.concat(
        [df_combined_pre_post_right, df_combined_pre_post_left], ignore_index=True
    )

    # Group multiple columns and reset the index
    df_group = (
        df_combined_averted_pre_post_gaze_only.groupby(
            ["GazeDirectionRight(X)", "GazeDirectionRight(Y)"]
        )
        .size()
        .reset_index(name="count")
    )
    # Count how many data that are under the same coordinate or spot
    pivot = df_group.pivot(
        index="GazeDirectionRight(X)", columns="GazeDirectionRight(Y)", values="count"
    )

    # Convert dataframe to numpy array
    pivot_array = pivot.to_numpy()

    # Plot the data with background picture

    fig, ax = plt.subplots(figsize=(7, 7))
    wd = matplotlib.cm.winter._segmentdata  # only has r,g,b
    wd["alpha"] = ((0.0, 0.0, 0.3), (0.3, 0.3, 1.0), (1.0, 1.0, 1.0))

    # modified colormap with changing alpha
    al_winter = LinearSegmentedColormap("AlphaWinter", wd)

    if tag == "averted":
        # get the overlay IMAGE as an array so we can plot it
        map_img = mpimg.imread("figures/averted2plot.png")
    elif tag == "direct":
        # get the overlay IMAGE as an array so we can plot it
        map_img = mpimg.imread("figures/direct2plot.png")
    elif tag == "natural":
        # get the overlay IMAGE as an array so we can plot it
        map_img = mpimg.imread("figures/natural2plot.png")

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

    # Make Capital the first letter of the tag
    tag_title = str.title(tag)
    # Combine as title of the figure
    tag_title = tag_title + "-Eye Gaze Condition"
    # Put as title
    ax.set_title(tag_title)

    ax.tick_params(axis="both", which="both", length=0)
    # Remove x labels
    ax.set(xticklabels=[], yticklabels=[])

    # Color bar related thing
    c_bar = hmax.collections[0].colorbar
    ticks_number = [ticks for ticks in c_bar.get_ticks()]
    c_bar.set_ticks([ticks_number[1], ticks_number[-2]])
    c_bar.set_ticklabels(["Low", "High"])

    # Save the figures
    if save_fig == True:

        fig = hmax.get_figure()

        # Save to the current working directory
        if path2savefigure == os.getcwd():
            cwd = os.getcwd()
            # os.chdir(cwd)
            file_name = tag + "_eye.png"
            cwd = os.path.join(cwd, file_name)
            fig.savefig(cwd, dpi=100)

        # Save to the desired path as mentioned in the parameter
        else:
            cwd = path2savefigure
            file_name = tag + "_eye.png"
            cwd = os.path.join(cwd, file_name)
            fig.savefig(cwd, dpi=100)


# %% Combined eye dataframe FIRST
# Initialize class of EyeAnalysis
eye_analysis = EyeAnalysis()
path2eyefiles = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/raw_cleaned_combined_experimental_eye_data/"
pre_tag = "natural_pre"
post_tag = "natural_post"
df_pre = eye_analysis.combine_eye_data(path2eyefiles, pre_tag)
df_post = eye_analysis.combine_eye_data(path2eyefiles, post_tag)


# %% Run plot heatmap function by using the above output of eye data

tag = "natural"
path2savefigure = "/hpc/igum002/codes/Hyperscanning2-redesign/phd_codes/figures"
plot_heatmap(
    df_pre[0],
    df_pre[1],
    df_post[0],
    df_post[1],
    tag,
    save_fig=True,
    path2savefigure=path2savefigure,
)

# %%
