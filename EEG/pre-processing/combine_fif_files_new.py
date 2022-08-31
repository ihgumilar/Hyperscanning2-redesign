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
# Import relevant packages
import mne
import tqdm
from tqdm import tqdm
import os

# %% [markdown]
# ### Baseline data

# %% [markdown]
# #### Define a directory where we store raw eeg files (*.fif), which are not combined yet (Baseline data)

# %%
# Go to a directory that stores raw fif file (not combined files)
raw_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/"
os.chdir(raw_dir)

# %% [markdown]
# #### Combine pre averted baseline

# %%

for i in tqdm(range(16), desc="Combining pre averted..."):  # type: ignore
    # Pre-averted
    if i < 9:

        averted_pre_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False
        )
        averted_pre_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S0"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_pre_files_to_combine = [ averted_pre_left_odd_subject, averted_pre_right_odd_subject]

            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S0"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_averted_files.save( # type: ignore
            combined_pre_averted_files_label, overwrite=True
        )

    else:

        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_pre_files_to_combine = [ averted_pre_left_odd_subject, averted_pre_right_odd_subject]

            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_averted_files.save( # type: ignore
            combined_pre_averted_files_label, overwrite=True)

# %% [markdown]
# #### Combine post averted baseline

# %%

for i in tqdm(range(16), desc="Combining post averted..."):  # type: ignore
    # post-averted
    if i < 9:

        averted_post_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_post_right_point_raw.fif", verbose=False
        )
        averted_post_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_post_files_to_combine = [
                averted_post_right_odd_subject,
                averted_post_left_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S0"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_post_files_to_combine = [ averted_post_left_odd_subject, averted_post_right_odd_subject]

            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S0"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )  

        combined_post_averted_files.save( # type: ignore
            combined_post_averted_files_label, overwrite=True
        )

    else:

        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_post_files_to_combine = [
                averted_post_right_odd_subject,
                averted_post_left_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_post_files_to_combine = [ averted_post_left_odd_subject, averted_post_right_odd_subject]

            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
                + "S"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )  

        combined_post_averted_files.save( # type: ignore
            combined_post_averted_files_label, overwrite=True)

# %% [markdown]
# ### Experimental

# %% [markdown]
# #### Combine pre averted experimental

# %%
# Go to a directory that stores raw fif file (not combined files)
raw_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/"
os.chdir(raw_dir)

# %%

for i in tqdm(range(16), desc="Combining pre averted..."):  # type: ignore
    # pre-averted
    if i < 9:

        averted_pre_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False
        )
        averted_pre_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_pre_files_to_combine = [ averted_pre_left_odd_subject, averted_pre_right_odd_subject]

            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_averted_files.save( # type: ignore
            combined_pre_averted_files_label, overwrite=True
        )

    else:

        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_pre_files_to_combine = [ averted_pre_left_odd_subject, averted_pre_right_odd_subject]

            combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)

            combined_pre_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-averted_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_averted_files.save( # type: ignore
            combined_pre_averted_files_label, overwrite=True)

# %% [markdown]
# #### Combine post averted experimental

# %%

for i in tqdm(range(16), desc="Combining post averted..."):  # type: ignore
    # post-averted
    if i < 9:

        averted_post_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_post_right_point_raw.fif", verbose=False
        )
        averted_post_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_post_files_to_combine = [
                averted_post_right_odd_subject,
                averted_post_left_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_post_files_to_combine = [ averted_post_left_odd_subject, averted_post_right_odd_subject]

            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )  

        combined_post_averted_files.save( # type: ignore
            combined_post_averted_files_label, overwrite=True
        )

    else:

        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            averted_post_files_to_combine = [
                averted_post_right_odd_subject,
                averted_post_left_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-averted_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            averted_post_files_to_combine = [ averted_post_left_odd_subject, averted_post_right_odd_subject]

            combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)

            combined_post_averted_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )  

        combined_post_averted_files.save( # type: ignore
            combined_post_averted_files_label, overwrite=True)

# %% [markdown]
# #### Combine pre direct experimental

# %%

for i in tqdm(range(16), desc="Combining pre direct..."):  # type: ignore
    # pre-direct
    if i < 9:

        direct_pre_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-direct_pre_right_point_raw.fif", verbose=False
        )
        direct_pre_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-direct_pre_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            direct_pre_files_to_combine = [
                direct_pre_right_odd_subject,
                direct_pre_left_odd_subject,
            ]
            combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)

            combined_pre_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-direct_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            direct_pre_files_to_combine = [ direct_pre_left_odd_subject, direct_pre_right_odd_subject]

            combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)

            combined_pre_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-direct_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_direct_files.save( # type: ignore
            combined_pre_direct_files_label, overwrite=True
        )

    else:

        direct_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-direct_pre_right_point_raw.fif", verbose=False)
        direct_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-direct_pre_left_point_raw.fif", verbose=False)
        direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
        combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            direct_pre_files_to_combine = [
                direct_pre_right_odd_subject,
                direct_pre_left_odd_subject,
            ]
            combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)

            combined_pre_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-direct_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            direct_pre_files_to_combine = [ direct_pre_left_odd_subject, direct_pre_right_odd_subject]

            combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)

            combined_pre_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-direct_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_direct_files.save( # type: ignore
            combined_pre_direct_files_label, overwrite=True)

# %% [markdown]
# #### Combine post direct experimental

# %%

for i in tqdm(range(16), desc="Combining post direct..."):  # type: ignore
    # post-direct
    if i < 9:

        direct_post_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-direct_post_right_point_raw.fif", verbose=False
        )
        direct_post_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-direct_post_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            direct_post_files_to_combine = [
                direct_post_right_odd_subject,
                direct_post_left_odd_subject,
            ]
            combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)

            combined_post_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-direct_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            direct_post_files_to_combine = [ direct_post_left_odd_subject, direct_post_right_odd_subject]

            combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)

            combined_post_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-direct_post_left_right_point_combined_raw.fif"
            )  

        combined_post_direct_files.save( # type: ignore
            combined_post_direct_files_label, overwrite=True
        )

    else:

        direct_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-direct_post_right_point_raw.fif", verbose=False)
        direct_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-direct_post_left_point_raw.fif", verbose=False)
        direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
        combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            direct_post_files_to_combine = [
                direct_post_right_odd_subject,
                direct_post_left_odd_subject,
            ]
            combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)

            combined_post_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-direct_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            direct_post_files_to_combine = [ direct_post_left_odd_subject, direct_post_right_odd_subject]

            combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)

            combined_post_direct_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-direct_post_left_right_point_combined_raw.fif"
            )  

        combined_post_direct_files.save( # type: ignore
            combined_post_direct_files_label, overwrite=True)

# %% [markdown]
# #### Combine pre natural experimental

# %%

for i in tqdm(range(16), desc="Combining pre natural..."):  # type: ignore
    # pre-natural
    if i < 9:

        natural_pre_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-natural_pre_right_point_raw.fif", verbose=False
        )
        natural_pre_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-natural_pre_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            natural_pre_files_to_combine = [
                natural_pre_right_odd_subject,
                natural_pre_left_odd_subject,
            ]
            combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)

            combined_pre_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-natural_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            natural_pre_files_to_combine = [ natural_pre_left_odd_subject, natural_pre_right_odd_subject]

            combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)

            combined_pre_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-natural_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_natural_files.save( # type: ignore
            combined_pre_natural_files_label, overwrite=True
        )

    else:

        natural_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-natural_pre_right_point_raw.fif", verbose=False)
        natural_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-natural_pre_left_point_raw.fif", verbose=False)
        natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
        combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            natural_pre_files_to_combine = [
                natural_pre_right_odd_subject,
                natural_pre_left_odd_subject,
            ]
            combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)

            combined_pre_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-natural_pre_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            natural_pre_files_to_combine = [ natural_pre_left_odd_subject, natural_pre_right_odd_subject]

            combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)

            combined_pre_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-natural_pre_left_right_point_combined_raw.fif"
            )  

        combined_pre_natural_files.save( # type: ignore
            combined_pre_natural_files_label, overwrite=True)

# %% [markdown]
# #### Combine post natural experimental

# %%

for i in tqdm(range(16), desc="Combining post natural..."):  # type: ignore
    # post-natural
    if i < 9:

        natural_post_right_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-natural_post_right_point_raw.fif", verbose=False
        )
        natural_post_left_odd_subject = mne.io.read_raw_fif(
            "EEG-S0" + str(i + 1) + "-natural_post_left_point_raw.fif", verbose=False
        )
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            natural_post_files_to_combine = [
                natural_post_right_odd_subject,
                natural_post_left_odd_subject,
            ]
            combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)

            combined_post_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-natural_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            natural_post_files_to_combine = [ natural_post_left_odd_subject, natural_post_right_odd_subject]

            combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)

            combined_post_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S0"
                + str(i + 1)
                + "-natural_post_left_right_point_combined_raw.fif"
            )  

        combined_post_natural_files.save( # type: ignore
            combined_post_natural_files_label, overwrite=True
        )

    else:

        natural_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-natural_post_right_point_raw.fif", verbose=False)
        natural_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i + 1) + "-natural_post_left_point_raw.fif", verbose=False)
        natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
        combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
        
        # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
        # For example, i = 0 actually takes S01 and keeps going...
        if (i % 2) == 0 :
        
            natural_post_files_to_combine = [
                natural_post_right_odd_subject,
                natural_post_left_odd_subject,
            ]
            combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)

            combined_post_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-natural_post_right_left_point_combined_raw.fif"
            )

        # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
        # For example, i = 1 actually takes S02 and keeps going...

        else:

            natural_post_files_to_combine = [ natural_post_left_odd_subject, natural_post_right_odd_subject]

            combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)

            combined_post_natural_files_label = (
                "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
                + "S"
                + str(i + 1)
                + "-natural_post_left_right_point_combined_raw.fif"
            )  

        combined_post_natural_files.save( # type: ignore
            combined_post_natural_files_label, overwrite=True)
