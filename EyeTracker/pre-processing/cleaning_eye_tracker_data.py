# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""
Delete parts of eye tracker data based on deleted epochs of EEG data.
Since we have extracted indices of deleted EEG epochs, then we use such indices
to delete some part of eye tracker data accordingly
"""
import os
import pickle
from os import listdir

import numpy as np
import pandas as pd

# %%
from pandas.api.types import is_numeric_dtype
from scipy import signal
from tqdm import tqdm


# %%
def delete_epoch_eye_tracker(
    eyetracker_data_path, path2save_cleaned_data, file_tag, indices
):
    # eyetracker_data_path = '/hpc/igum002/codes/frontiers_hyperscanning2/eye_tracker_data_combined/'
    # file_tag = "averted_pre"

    # Go to a folder where combined eye tracker data has been stored
    # os.chdir(eyetracker_data_path)

    # List specified files based on file_tag parameter that has been defined
    files = [file for file in os.listdir(eyetracker_data_path) if file_tag in file]
    files = sorted(files, key=lambda x: int(x.partition("S")[2].partition("-")[0]))
    begin = 0
    end = len(files)
    step = 2
    counter = 0

    for idx in tqdm(
        range(begin, end, step),
        desc="Have some coffee bro, while we are processing your files :) ",
    ):

        # df1 = pd.read_csv(eyetracker_data_path + files[0])
        df1 = pd.read_csv(eyetracker_data_path + files[idx])
        df2 = pd.read_csv(eyetracker_data_path + files[idx + 1])

        # Drop missing values
        df1 = df1.dropna()
        df2 = df2.dropna()
        dfNew1 = pd.DataFrame()
        dfNew2 = pd.DataFrame()

        # Resample the data (125 sampling rate x 120 seconds = 15000 rows)
        for column in df1.columns:
            if is_numeric_dtype(df1[column]):
                dfNew1[column] = signal.resample(df1[column].tolist(), 15000)

        for column in df2.columns:
            if is_numeric_dtype(df2[column]):
                dfNew2[column] = signal.resample(df2[column].tolist(), 15000)

        # Create dictionary which contains indices to delete
        key_counter = 0
        labels_indices = {}
        for val in range(0, 15000, 125):
            idx_start = val
            idx_end = val + 125
            labels_indices.update({key_counter: [idx_start, idx_end]})
            key_counter += 1
            labels_indices_2_delete = []

        if idx >= 1:
            counter += 1

        # TODO  Understand what this code does. Error when processing subject 11
        # NOTE Indices of deleted epochs of EEG, there are only 8 lists for now. Since there are 8 pairs.
        # Those indices are used for each pair of eye tracker data

        for individual_indices in indices[counter]:
            labels_indices_2_delete.append(labels_indices[individual_indices])

        # Mark with 1 for rows to be deleted
        for val in labels_indices_2_delete:
            begin_idx_delete = val[0]
            end_idx_delete = val[1]

            dfNew1.loc[dfNew1.index[begin_idx_delete:end_idx_delete], "Mark2Delete"] = 1
            dfNew2.loc[dfNew2.index[begin_idx_delete:end_idx_delete], "Mark2Delete"] = 1

            # Delete rows which have marked 1 in column Mark2Delete
            dfNew1 = dfNew1[dfNew1["Mark2Delete"] != 1]
            dfNew2 = dfNew2[dfNew2["Mark2Delete"] != 1]

            # Drop column of Mark2Delete
            dfNew1.drop("Mark2Delete", axis=1, inplace=True)
            dfNew2.drop("Mark2Delete", axis=1, inplace=True)

        # Save cleaned data
        # path2save_cleaned_data = '/hpc/igum002/codes/frontiers_hyperscanning2/eye_tracker_data_clean_new/'
        df1_name = files[idx]
        df2_name = files[idx + 1]
        dfNew1.to_csv(
            path2save_cleaned_data + df1_name[: df1_name.rfind(".")] + "_clean.csv"
        )
        dfNew2.to_csv(
            path2save_cleaned_data + df2_name[: df2_name.rfind(".")] + "_clean.csv"
        )
        print("Processed : " + df1_name)
        print("Processed : " + df2_name)

    print("The eye tracker files have been cleaned and you are good to go, Bro !")


# %% [markdown]
# ## Load deleted epochs indices (all eye conditions)

# %% Load the deleted indices of epochs of eeg data
# Change to a directory where combined eye tracker data are located
os.chdir(
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/deleted_epochs_indices/"
)

# averted pre
with open("list_deleted_epoch_indices_averted_pre.pkl", "rb") as handle:
    deleted_epochs_indices_averted_pre = pickle.load(handle)
# deleted_epochs_indices_averted_pre[0]
# averted post
with open("list_deleted_epoch_indices_averted_post.pkl", "rb") as handle:
    deleted_epochs_indices_averted_post = pickle.load(handle)

# direct pre
with open("list_deleted_epoch_indices_direct_pre.pkl", "rb") as handle:
    deleted_epochs_indices_direct_pre = pickle.load(handle)

# direct post
with open("list_deleted_epoch_indices_direct_post.pkl", "rb") as handle:
    deleted_epochs_indices_direct_post = pickle.load(handle)

# natural pre
with open("list_deleted_epoch_indices_natural_pre.pkl", "rb") as handle:
    deleted_epochs_indices_natural_pre = pickle.load(handle)

# natural post
with open("list_deleted_epoch_indices_natural_post.pkl", "rb") as handle:
    deleted_epochs_indices_natural_post = pickle.load(handle)

# %% [markdown]
# ### Clean eye tracker data :
# #### Delete part of eye tracker data where the same part of EEG data (epochs) has been removed

# %%
# def delete_epoch_eye_tracker(eyetracker_data_path, path2save_cleaned_data, file_tag, indices):

eyetracker_data_path = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/"
path2save_cleaned_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/raw_cleaned_combined_experimental_eye_data/"
averted_pre_tag = "averted_pre"
# averted_post_tag = "averted_post"
# direct_pre_tag = "direct_pre"
# direct_post_tag = "direct_post"
# natural_pre_tag = "natural_pre"
# natural_post_tag = "natural_post"

# Clean eye tracker data
delete_epoch_eye_tracker(
    eyetracker_data_path,
    path2save_cleaned_data,
    averted_pre_tag,
    deleted_epochs_indices_averted_pre,
)


# %% Delete the same epochs (data) that has been deleted in EEG in eye tracker data files

# averted pre
delete_epoch_eye_tracker("averted_pre", deleted_epochs_indices_averted_pre)
# averted post
delete_epoch_eye_tracker("averted_post", deleted_epochs_indices_averted_post)
# direct pre
delete_epoch_eye_tracker("direct_pre", deleted_epochs_indices_direct_pre)
# direct post
delete_epoch_eye_tracker("direct_post", deleted_epochs_indices_direct_post)
# natural pre
delete_epoch_eye_tracker("natural_pre", deleted_epochs_indices_natural_pre)
# natural post
delete_epoch_eye_tracker("natural_post", deleted_epochs_indices_natural_post)
