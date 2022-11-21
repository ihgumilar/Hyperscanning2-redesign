# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
import os

# %%
# Import relevant packages
import mne
import pandas as pd
import tqdm
from tqdm import tqdm

# %% [markdown]
# ### Baseline data

# %% [markdown]
# #### Define some directories :
# *  Where we store raw eeg files (*.csv), which are not combined yet (Baseline data)
# *  Where we would like to store combined eye tracker data (Baseline data)
#
#

# %%
# Raw baseline data (csv file) (not combined yet)
raw_dir_baseline = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_baseline_eye_data/"
)

# Folder to store combined eye tracker data (baseline data)
raw_combined_baseline_data_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_baseline_eye_data/raw_combined_baseline_eye_data/"

# Change to directory which stores raw baseline data (not combined)
os.chdir(raw_dir_baseline)

# %% [markdown]
# #### Combine pre averted baseline

# %%
for i in tqdm(range(15), desc="Combining pre averted..."):  # type: ignore
    # Pre-averted (for processing subject 1 - 9)
    if i < 9:

        # Load averted pre right
        averted_pre_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_pre_right_point_raw.csv"
        )
        averted_pre_right_odd_subject = pd.read_csv(
            averted_pre_right_odd_subject_file_name
        )

        # Load Load averted pre left
        averted_pre_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_pre_left_point_raw.csv"
        )
        averted_pre_left_odd_subject = pd.read_csv(
            averted_pre_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

    # Pre-averted (for processing subject 10 onwards)
    else:
        # Load averted pre right
        averted_pre_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_pre_right_point_raw.csv"
        )
        averted_pre_right_odd_subject = pd.read_csv(
            averted_pre_right_odd_subject_file_name
        )

        # Load Load averted pre left
        averted_pre_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_pre_left_point_raw.csv"
        )
        averted_pre_left_odd_subject = pd.read_csv(
            averted_pre_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)


# %% [markdown]
# #### Combine post averted baseline

# %%
for i in tqdm(range(15), desc="Combining post averted..."):  # type: ignore
    # Pre-averted (for processing subject 1 - 9)
    if i < 9:

        # Load averted post right
        averted_post_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_post_right_point_raw.csv"
        )
        averted_post_right_odd_subject = pd.read_csv(
            averted_post_right_odd_subject_file_name
        )

        # Load Load averted post left
        averted_post_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_post_left_point_raw.csv"
        )
        averted_post_left_odd_subject = pd.read_csv(
            averted_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_right_odd_subject, averted_post_left_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_left_odd_subject, averted_post_right_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)

    # Pre-averted (for processing subject 10 onwards)
    else:
        # Load averted post right
        averted_post_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_post_right_point_raw.csv"
        )
        averted_post_right_odd_subject = pd.read_csv(
            averted_post_right_odd_subject_file_name
        )

        # Load Load averted post left
        averted_post_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_post_left_point_raw.csv"
        )
        averted_post_left_odd_subject = pd.read_csv(
            averted_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_right_odd_subject, averted_post_left_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_left_odd_subject, averted_post_right_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_baseline_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)


# %% [markdown]
# ### Experimental

# %% [markdown]
# #### Define some directories :
# *  Where we store raw eeg files (*.csv), which are not combined yet (Experimental data)
# *  Where we would like to store combined eye tracker data (Experimental data)
#
#

# %%
# Raw experimental data (csv file) (not combined)
raw_dir_experimental = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/"

# Folder to store combined eye tracker data (experimental data)
raw_combined_experimental_data_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/"

# Change to directory which stores raw experimental data (not combined)
os.chdir(raw_dir_experimental)

# %%
os.getcwd()
os.listdir("./raw_combined_experimental_data_directory")

# %% [markdown]
# #### Combine pre averted experimental

# %%
for i in tqdm(range(15), desc="Combining pre averted..."):  # type: ignore
    # Pre-averted (for processing subject 1 - 9)
    if i < 9:

        # Load averted pre right
        averted_pre_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_pre_right_point_raw.csv"
        )
        averted_pre_right_odd_subject = pd.read_csv(
            averted_pre_right_odd_subject_file_name
        )

        # Load Load averted pre left
        averted_pre_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_pre_left_point_raw.csv"
        )
        averted_pre_left_odd_subject = pd.read_csv(
            averted_pre_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

    # Pre-averted (for processing subject 10 onwards)
    else:
        # Load averted pre right
        averted_pre_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_pre_right_point_raw.csv"
        )
        averted_pre_right_odd_subject = pd.read_csv(
            averted_pre_right_odd_subject_file_name
        )

        # Load Load averted pre left
        averted_pre_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_pre_left_point_raw.csv"
        )
        averted_pre_left_odd_subject = pd.read_csv(
            averted_pre_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_averted_files = pd.concat(
                [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-averted baseline
            combined_pre_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-averted baseline file to csv
            combined_pre_averted_files.to_csv(combined_pre_averted_files_label)


# %%
os.getcwd()

# %% [markdown]
# #### Combine post averted experimental

# %%
for i in tqdm(range(15), desc="Combining post averted..."):  # type: ignore
    # post-averted (for processing subject 1 - 9)
    if i < 9:

        # Load averted post right
        averted_post_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_post_right_point_raw.csv"
        )
        averted_post_right_odd_subject = pd.read_csv(
            averted_post_right_odd_subject_file_name
        )

        # Load Load averted post left
        averted_post_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-averted_post_left_point_raw.csv"
        )
        averted_post_left_odd_subject = pd.read_csv(
            averted_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_right_odd_subject, averted_post_left_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_left_odd_subject, averted_post_right_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)

    # post-averted (for processing subject 10 onwards)
    else:
        # Load averted post right
        averted_post_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_post_right_point_raw.csv"
        )
        averted_post_right_odd_subject = pd.read_csv(
            averted_post_right_odd_subject_file_name
        )

        # Load Load averted post left
        averted_post_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-averted_post_left_point_raw.csv"
        )
        averted_post_left_odd_subject = pd.read_csv(
            averted_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_right_odd_subject, averted_post_left_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_averted_files = pd.concat(
                [averted_post_left_odd_subject, averted_post_right_odd_subject]
            )

            # Create  file name for combine files of post-averted baseline
            combined_post_averted_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-averted baseline file to csv
            combined_post_averted_files.to_csv(combined_post_averted_files_label)


# %% [markdown]
# #### Combine pre direct experimental

# %%
for i in tqdm(range(15), desc="Combining pre direct..."):  # type: ignore
    # Pre-direct (for processing subject 1 - 9)
    if i < 9:

        # Load direct pre right
        direct_pre_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-direct_pre_right_point_raw.csv"
        )
        direct_pre_right_odd_subject = pd.read_csv(
            direct_pre_right_odd_subject_file_name
        )

        # Load Load direct pre left
        direct_pre_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-direct_pre_left_point_raw.csv"
        )
        direct_pre_left_odd_subject = pd.read_csv(direct_pre_left_odd_subject_file_name)

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_direct_files = pd.concat(
                [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-direct baseline
            combined_pre_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-direct_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-direct baseline file to csv
            combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_direct_files = pd.concat(
                [direct_pre_left_odd_subject, direct_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-direct baseline
            combined_pre_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-direct_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-direct baseline file to csv
            combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

    # Pre-direct (for processing subject 10 onwards)
    else:
        # Load direct pre right
        direct_pre_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-direct_pre_right_point_raw.csv"
        )
        direct_pre_right_odd_subject = pd.read_csv(
            direct_pre_right_odd_subject_file_name
        )

        # Load Load direct pre left
        direct_pre_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-direct_pre_left_point_raw.csv"
        )
        direct_pre_left_odd_subject = pd.read_csv(direct_pre_left_odd_subject_file_name)

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_direct_files = pd.concat(
                [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-direct baseline
            combined_pre_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-direct_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-direct baseline file to csv
            combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_direct_files = pd.concat(
                [direct_pre_left_odd_subject, direct_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-direct baseline
            combined_pre_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-direct_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-direct baseline file to csv
            combined_pre_direct_files.to_csv(combined_pre_direct_files_label)


# %% [markdown]
# #### Combine post direct experimental

# %%
for i in tqdm(range(15), desc="Combining post direct..."):  # type: ignore
    # post-direct (for processing subject 1 - 9)
    if i < 9:

        # Load direct post right
        direct_post_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-direct_post_right_point_raw.csv"
        )
        direct_post_right_odd_subject = pd.read_csv(
            direct_post_right_odd_subject_file_name
        )

        # Load Load direct post left
        direct_post_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-direct_post_left_point_raw.csv"
        )
        direct_post_left_odd_subject = pd.read_csv(
            direct_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_direct_files = pd.concat(
                [direct_post_right_odd_subject, direct_post_left_odd_subject]
            )

            # Create  file name for combine files of post-direct baseline
            combined_post_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-direct_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-direct baseline file to csv
            combined_post_direct_files.to_csv(combined_post_direct_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_direct_files = pd.concat(
                [direct_post_left_odd_subject, direct_post_right_odd_subject]
            )

            # Create  file name for combine files of post-direct baseline
            combined_post_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-direct_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-direct baseline file to csv
            combined_post_direct_files.to_csv(combined_post_direct_files_label)

    # post-direct (for processing subject 10 onwards)
    else:
        # Load direct post right
        direct_post_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-direct_post_right_point_raw.csv"
        )
        direct_post_right_odd_subject = pd.read_csv(
            direct_post_right_odd_subject_file_name
        )

        # Load Load direct post left
        direct_post_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-direct_post_left_point_raw.csv"
        )
        direct_post_left_odd_subject = pd.read_csv(
            direct_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_direct_files = pd.concat(
                [direct_post_right_odd_subject, direct_post_left_odd_subject]
            )

            # Create  file name for combine files of post-direct baseline
            combined_post_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-direct_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-direct baseline file to csv
            combined_post_direct_files.to_csv(combined_post_direct_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_direct_files = pd.concat(
                [direct_post_left_odd_subject, direct_post_right_odd_subject]
            )

            # Create  file name for combine files of post-direct baseline
            combined_post_direct_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-direct_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-direct baseline file to csv
            combined_post_direct_files.to_csv(combined_post_direct_files_label)


# %% [markdown]
# #### Combine pre natural experimental

# %%
for i in tqdm(range(15), desc="Combining pre natural..."):  # type: ignore
    # Pre-natural (for processing subject 1 - 9)
    if i < 9:

        # Load natural pre right
        natural_pre_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-natural_pre_right_point_raw.csv"
        )
        natural_pre_right_odd_subject = pd.read_csv(
            natural_pre_right_odd_subject_file_name
        )

        # Load Load natural pre left
        natural_pre_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-natural_pre_left_point_raw.csv"
        )
        natural_pre_left_odd_subject = pd.read_csv(
            natural_pre_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_natural_files = pd.concat(
                [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-natural baseline
            combined_pre_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-natural_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-natural baseline file to csv
            combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_natural_files = pd.concat(
                [natural_pre_left_odd_subject, natural_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-natural baseline
            combined_pre_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-natural_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-natural baseline file to csv
            combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

    # Pre-natural (for processing subject 10 onwards)
    else:
        # Load natural pre right
        natural_pre_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-natural_pre_right_point_raw.csv"
        )
        natural_pre_right_odd_subject = pd.read_csv(
            natural_pre_right_odd_subject_file_name
        )

        # Load Load natural pre left
        natural_pre_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-natural_pre_left_point_raw.csv"
        )
        natural_pre_left_odd_subject = pd.read_csv(
            natural_pre_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_pre_natural_files = pd.concat(
                [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
            )

            # Create  file name for combine files of pre-natural baseline
            combined_pre_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-natural_pre_right_left_point_combined_raw.csv"
            )

            # Save combine pre-natural baseline file to csv
            combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_pre_natural_files = pd.concat(
                [natural_pre_left_odd_subject, natural_pre_right_odd_subject]
            )

            # Create  file name for combine files of pre-natural baseline
            combined_pre_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-natural_pre_left_right_point_combined_raw.csv"
            )

            # Save combine pre-natural baseline file to csv
            combined_pre_natural_files.to_csv(combined_pre_natural_files_label)


# %% [markdown]
# #### Combine post natural experimental

# %%
for i in tqdm(range(15), desc="Combining post natural..."):  # type: ignore
    # post-natural (for processing subject 1 - 9)
    if i < 9:

        # Load natural post right
        natural_post_right_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-natural_post_right_point_raw.csv"
        )
        natural_post_right_odd_subject = pd.read_csv(
            natural_post_right_odd_subject_file_name
        )

        # Load Load natural post left
        natural_post_left_odd_subject_file_name = (
            "EyeTracker-S0" + str(i + 1) + "-natural_post_left_point_raw.csv"
        )
        natural_post_left_odd_subject = pd.read_csv(
            natural_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_natural_files = pd.concat(
                [natural_post_right_odd_subject, natural_post_left_odd_subject]
            )

            # Create  file name for combine files of post-natural baseline
            combined_post_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-natural_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-natural baseline file to csv
            combined_post_natural_files.to_csv(combined_post_natural_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_natural_files = pd.concat(
                [natural_post_left_odd_subject, natural_post_right_odd_subject]
            )

            # Create  file name for combine files of post-natural baseline
            combined_post_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S0"
                + str(i + 1)
                + "-natural_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-natural baseline file to csv
            combined_post_natural_files.to_csv(combined_post_natural_files_label)

    # post-natural (for processing subject 10 onwards)
    else:
        # Load natural post right
        natural_post_right_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-natural_post_right_point_raw.csv"
        )
        natural_post_right_odd_subject = pd.read_csv(
            natural_post_right_odd_subject_file_name
        )

        # Load Load natural post left
        natural_post_left_odd_subject_file_name = (
            "EyeTracker-S" + str(i + 1) + "-natural_post_left_point_raw.csv"
        )
        natural_post_left_odd_subject = pd.read_csv(
            natural_post_left_odd_subject_file_name
        )

        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0:

            # Combine RIGHT => LEFT hand data
            combined_post_natural_files = pd.concat(
                [natural_post_right_odd_subject, natural_post_left_odd_subject]
            )

            # Create  file name for combine files of post-natural baseline
            combined_post_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-natural_post_right_left_point_combined_raw.csv"
            )

            # Save combine post-natural baseline file to csv
            combined_post_natural_files.to_csv(combined_post_natural_files_label)

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...
        else:

            # Combine LEFT => RIGHT hand data
            combined_post_natural_files = pd.concat(
                [natural_post_left_odd_subject, natural_post_right_odd_subject]
            )

            # Create  file name for combine files of post-natural baseline
            combined_post_natural_files_label = (
                raw_combined_experimental_data_directory
                + "EyeTracker-S"
                + str(i + 1)
                + "-natural_post_left_right_point_combined_raw.csv"
            )

            # Save combine post-natural baseline file to csv
            combined_post_natural_files.to_csv(combined_post_natural_files_label)
