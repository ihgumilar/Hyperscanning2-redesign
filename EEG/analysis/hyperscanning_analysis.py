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
import csv
import io
import os
import pickle
import statistics
import warnings
from collections import OrderedDict
from copy import copy
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import requests
import scipy

# import scipy.stats as stats
from hypyp import analyses, prep, stats, viz
from hypyp.ext.mpl3d import glm
from hypyp.ext.mpl3d.camera import Camera
from hypyp.ext.mpl3d.mesh import Mesh
from icecream import ic
from mpl_toolkits.mplot3d import Axes3D

# from scipy.stats import chi2, chi2_contingency
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Function to count how long processing files for each eye condition

# %% Time conversion
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# %% markdown [markdown]
# ## Direct eye(Pre - training)
# %%
# Container for no. of connections of ALL participants
path_2_experimental_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
path_2_dir_2_save_preprocessed_data = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
)
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_direct_pre_suffix = "-direct_pre_right_left_point_combined_raw.fif"
even_subject_direct_pre_suffix = "-direct_pre_left_right_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_direct_pre = []

total_n_connections_all_pairs_direct_pre = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_direct_pre = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_direct_pre = []

list_circular_correlation_direct_pre_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_experimental_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_direct = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_direct_pre_suffix
        )
        fname2_direct = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_direct_pre_suffix
        )
        # Replace fname2_direct variable
        if (i + 2) == 10:
            fname2_direct = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_direct_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_direct_pre_suffix
        )
        fname2_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_direct_pre_suffix
        )
        # Replace fname2_direct variable
        if (i + 2) == 20:
            fname2_direct = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_direct_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_direct_pre_suffix
        )
        fname2_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_direct_pre_suffix
        )
        # Replace fname2_direct variable
        if (i + 2) == 30:
            fname2_direct = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_direct_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_direct = fname1_direct
    fname_S2_direct = fname2_direct

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_direct = mne.io.read_raw_fif(fname_S1_direct, preload=True, verbose=False)
    raw2_direct = mne.io.read_raw_fif(fname_S2_direct, preload=True, verbose=False)

    raw1_direct.info["bads"] = original_bad_channels1
    raw2_direct.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_direct = raw1_direct.filter(l_freq=1, h_freq=40)
    raw2_direct = raw2_direct.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_direct = raw1_direct.copy().interpolate_bads(reset_bads=True)
    raw2_direct = raw2_direct.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_direct, id=1, duration=1)

    # Epoch length is 1 second
    epo1_direct = mne.Epochs(
        raw1_direct,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_direct = mne.Epochs(
        raw2_direct,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_direct, epo2_direct])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_direct.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_direct, epo2_direct],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_direct, epo2_direct], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_direct_pre.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_direct_pre_no_filter_all.append(theta)
    list_circular_correlation_direct_pre_no_filter_all.append(alpha)
    list_circular_correlation_direct_pre_no_filter_all.append(beta)
    list_circular_correlation_direct_pre_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-direct) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

  
    # Extract file name from path (subject 1)
    preproc_filename1 = fname_S1_direct
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_direct.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_direct
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_direct.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"


    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)


# Change to a directory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_direct_pre_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_direct_pre_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-direct) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_direct_pre.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_direct_pre, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
print("(pre-direct) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% markdown [markdown]
# ## Direct eye(Post - training)
# %%
# Container for no. of connections of ALL participants
path_2_experimental_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
path_2_dir_2_save_preprocessed_data = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
)
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_direct_post_suffix = "-direct_post_left_right_point_combined_raw.fif"
even_subject_direct_post_suffix = "-direct_post_right_left_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_direct_post = []

total_n_connections_all_pairs_direct_post = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_direct_post = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_direct_post = []

list_circular_correlation_direct_post_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_experimental_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_direct = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_direct_post_suffix
        )
        fname2_direct = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_direct_post_suffix
        )
        # Replace fname2_direct variable
        if (i + 2) == 10:
            fname2_direct = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_direct_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_direct_post_suffix
        )
        fname2_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_direct_post_suffix
        )
        # Replace fname2_direct variable
        if (i + 2) == 20:
            fname2_direct = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_direct_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_direct_post_suffix
        )
        fname2_direct = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_direct_post_suffix
        )
        # Replace fname2_direct variable
        if (i + 2) == 30:
            fname2_direct = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_direct_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_direct = fname1_direct
    fname_S2_direct = fname2_direct

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_direct = mne.io.read_raw_fif(fname_S1_direct, preload=True, verbose=False)
    raw2_direct = mne.io.read_raw_fif(fname_S2_direct, preload=True, verbose=False)

    raw1_direct.info["bads"] = original_bad_channels1
    raw2_direct.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_direct = raw1_direct.filter(l_freq=1, h_freq=40)
    raw2_direct = raw2_direct.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_direct = raw1_direct.copy().interpolate_bads(reset_bads=True)
    raw2_direct = raw2_direct.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_direct, id=1, duration=1)

    # Epoch length is 1 second
    epo1_direct = mne.Epochs(
        raw1_direct,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_direct = mne.Epochs(
        raw2_direct,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_direct, epo2_direct])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_direct.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_direct, epo2_direct],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_direct, epo2_direct], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_direct_post.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_direct_post_no_filter_all.append(theta)
    list_circular_correlation_direct_post_no_filter_all.append(alpha)
    list_circular_correlation_direct_post_no_filter_all.append(beta)
    list_circular_correlation_direct_post_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-direct) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    # Extract file name from path (subject 1)
    preproc_filename1 = fname_S1_direct
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_direct.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_direct
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_direct.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)
    

# Change to a directory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_direct_post_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_direct_post_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-direct) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_direct_post.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_direct_post, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
print("(pre-direct) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% markdown [markdown]
# ## Averted eye(Pre - training)

# %%
# Container for no. of connections of ALL participants
path_2_experimental_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
path_2_dir_2_save_preprocessed_data = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
)
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_averted_pre_suffix = "-averted_pre_right_left_point_combined_raw.fif"
even_subject_averted_pre_suffix = "-averted_pre_left_right_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_averted_pre = []

total_n_connections_all_pairs_averted_pre = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_averted_pre = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_averted_pre = []

list_circular_correlation_averted_pre_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_experimental_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_averted = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_averted_pre_suffix
        )
        fname2_averted = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_averted_pre_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 10:
            fname2_averted = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_pre_suffix
        )
        fname2_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_pre_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 20:
            fname2_averted = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_pre_suffix
        )
        fname2_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_pre_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 30:
            fname2_averted = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40]
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_averted = fname1_averted
    fname_S2_averted = fname2_averted

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_averted = mne.io.read_raw_fif(fname_S1_averted, preload=True, verbose=False)
    raw2_averted = mne.io.read_raw_fif(fname_S2_averted, preload=True, verbose=False)

    raw1_averted.info["bads"] = original_bad_channels1
    raw2_averted.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_averted = raw1_averted.filter(l_freq=1, h_freq=40)
    raw2_averted = raw2_averted.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_averted = raw1_averted.copy().interpolate_bads(reset_bads=True)
    raw2_averted = raw2_averted.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_averted, id=1, duration=1)

    # Epoch length is 1 second
    epo1_averted = mne.Epochs(
        raw1_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_averted = mne.Epochs(
        raw2_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_averted, epo2_averted])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_averted.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_averted, epo2_averted],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_averted, epo2_averted], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_averted_pre.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_averted_pre_no_filter_all.append(theta)
    list_circular_correlation_averted_pre_no_filter_all.append(alpha)
    list_circular_correlation_averted_pre_no_filter_all.append(beta)
    list_circular_correlation_averted_pre_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-averted) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    # Extract file name from path (subject 1)
    preproc_filename1 = fname_S1_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_averted.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_averted.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)

# Change to a a directory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_averted_pre_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_averted_pre_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-averted) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_averted_pre.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_averted_pre, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
print("(pre-averted) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% markdown [markdown]
# ## Averted eye(Post - training)
# %%
# Container for no. of connections of ALL participants
path_2_experimental_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
path_2_dir_2_save_preprocessed_data = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
)
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_averted_post_suffix = "-averted_post_left_right_point_combined_raw.fif"
even_subject_averted_post_suffix = "-averted_post_right_left_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_averted_post = []

total_n_connections_all_pairs_averted_post = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_averted_post = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_averted_post = []

list_circular_correlation_averted_post_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_experimental_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_averted = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_averted_post_suffix
        )
        fname2_averted = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_averted_post_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 10:
            fname2_averted = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_post_suffix
        )
        fname2_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_post_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 20:
            fname2_averted = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_post_suffix
        )
        fname2_averted = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_post_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 30:
            fname2_averted = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_averted = fname1_averted
    fname_S2_averted = fname2_averted

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_averted = mne.io.read_raw_fif(fname_S1_averted, preload=True, verbose=False)
    raw2_averted = mne.io.read_raw_fif(fname_S2_averted, preload=True, verbose=False)

    raw1_averted.info["bads"] = original_bad_channels1
    raw2_averted.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_averted = raw1_averted.filter(l_freq=1, h_freq=40)
    raw2_averted = raw2_averted.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_averted = raw1_averted.copy().interpolate_bads(reset_bads=True)
    raw2_averted = raw2_averted.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_averted, id=1, duration=1)

    # Epoch length is 1 second
    epo1_averted = mne.Epochs(
        raw1_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_averted = mne.Epochs(
        raw2_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_averted, epo2_averted])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_averted.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_averted, epo2_averted],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_averted, epo2_averted], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_averted_post.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_averted_post_no_filter_all.append(theta)
    list_circular_correlation_averted_post_no_filter_all.append(alpha)
    list_circular_correlation_averted_post_no_filter_all.append(beta)
    list_circular_correlation_averted_post_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-averted) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    # Extract file name from path (subject 1)
    preproc_filename1 = fname_S1_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_averted.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_averted.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

     # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)

# Change to a directory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_averted_post_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_averted_post_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-averted) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_averted_post.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_averted_post,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
print("(pre-averted) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")


# %% markdown [markdown]
# ## Natural eye(Pre - training)

# %%
# Container for no. of connections of ALL participants
path_2_experimental_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
path_2_dir_2_save_preprocessed_data = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
)
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_natural_pre_suffix = "-natural_pre_right_left_point_combined_raw.fif"
even_subject_natural_pre_suffix = "-natural_pre_left_right_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_natural_pre = []

total_n_connections_all_pairs_natural_pre = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_natural_pre = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_natural_pre = []

list_circular_correlation_natural_pre_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_experimental_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_natural = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_natural_pre_suffix
        )
        fname2_natural = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_natural_pre_suffix
        )
        # Replace fname2_natural variable
        if (i + 2) == 10:
            fname2_natural = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_natural_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_natural_pre_suffix
        )
        fname2_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_natural_pre_suffix
        )
        # Replace fname2_natural variable
        if (i + 2) == 20:
            fname2_natural = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_natural_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_natural_pre_suffix
        )
        fname2_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_natural_pre_suffix
        )
        # Replace fname2_natural variable
        if (i + 2) == 30:
            fname2_natural = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_natural_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_natural = fname1_natural
    fname_S2_natural = fname2_natural

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_natural = mne.io.read_raw_fif(fname_S1_natural, preload=True, verbose=False)
    raw2_natural = mne.io.read_raw_fif(fname_S2_natural, preload=True, verbose=False)

    raw1_natural.info["bads"] = original_bad_channels1
    raw2_natural.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_natural = raw1_natural.filter(l_freq=1, h_freq=40)
    raw2_natural = raw2_natural.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_natural = raw1_natural.copy().interpolate_bads(reset_bads=True)
    raw2_natural = raw2_natural.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_natural, id=1, duration=1)

    # Epoch length is 1 second
    epo1_natural = mne.Epochs(
        raw1_natural,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_natural = mne.Epochs(
        raw2_natural,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_natural, epo2_natural])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_natural.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_natural, epo2_natural],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_natural, epo2_natural], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_natural_pre.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_natural_pre_no_filter_all.append(theta)
    list_circular_correlation_natural_pre_no_filter_all.append(alpha)
    list_circular_correlation_natural_pre_no_filter_all.append(beta)
    list_circular_correlation_natural_pre_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-natural) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    preproc_filename1 = fname_S1_natural
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_natural.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_natural
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_natural.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)

# Change to a a directory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_natural_pre_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_natural_pre_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-natural) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

    
# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_natural_pre.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_natural_pre, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
print("(pre-natural) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")


# %% markdown [markdown]
# ## Natural eye(Post - training)
# %%
# Container for no. of connections of ALL participants
path_2_experimental_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
path_2_dir_2_save_preprocessed_data = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
)
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_natural_post_suffix = "-natural_post_left_right_point_combined_raw.fif"
even_subject_natural_post_suffix = "-natural_post_right_left_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_natural_post = []

total_n_connections_all_pairs_natural_post = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_natural_post = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_natural_post = []

list_circular_correlation_natural_post_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_experimental_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_natural = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_natural_post_suffix
        )
        fname2_natural = (
            path_2_experimental_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_natural_post_suffix
        )
        # Replace fname2_natural variable
        if (i + 2) == 10:
            fname2_natural = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_natural_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_natural_post_suffix
        )
        fname2_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_natural_post_suffix
        )
        # Replace fname2_natural variable
        if (i + 2) == 20:
            fname2_natural = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_natural_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_natural_post_suffix
        )
        fname2_natural = (
            path_2_experimental_data_dir
            + "S"
            + str(i + 2)
            + even_subject_natural_post_suffix
        )
        # Replace fname2_natural variable
        if (i + 2) == 30:
            fname2_natural = (
                path_2_experimental_data_dir
                + "S"
                + str(i + 2)
                + even_subject_natural_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_natural = fname1_natural
    fname_S2_natural = fname2_natural

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_natural = mne.io.read_raw_fif(fname_S1_natural, preload=True, verbose=False)
    raw2_natural = mne.io.read_raw_fif(fname_S2_natural, preload=True, verbose=False)

    raw1_natural.info["bads"] = original_bad_channels1
    raw2_natural.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_natural = raw1_natural.filter(l_freq=1, h_freq=40)
    raw2_natural = raw2_natural.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_natural = raw1_natural.copy().interpolate_bads(reset_bads=True)
    raw2_natural = raw2_natural.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_natural, id=1, duration=1)

    # Epoch length is 1 second
    epo1_natural = mne.Epochs(
        raw1_natural,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_natural = mne.Epochs(
        raw2_natural,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_natural, epo2_natural])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_natural.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_natural, epo2_natural],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_natural, epo2_natural], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_natural_post.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_natural_post_no_filter_all.append(theta)
    list_circular_correlation_natural_post_no_filter_all.append(alpha)
    list_circular_correlation_natural_post_no_filter_all.append(beta)
    list_circular_correlation_natural_post_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-natural) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    preproc_filename1 = fname_S1_natural
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_natural.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_natural
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_natural.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)

# Change to a directory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_natural_post_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_natural_post_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-natural) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_natural_post.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_natural_post,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
print("(pre-natural) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")


# %% [markdown]
# ## Baseline Averted pre

# %%
# Container for no. of connections of ALL participants
path_2_baseline_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
path_2_dir_2_save_preprocessed_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_averted_baseline_pre_suffix = "-averted_pre_right_left_point_combined_raw.fif"
even_subject_averted_baseline_pre_suffix = "-averted_pre_left_right_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_averted_baseline_pre = []

total_n_connections_all_pairs_averted_baseline_pre = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_averted_baseline_pre = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_averted_baseline_pre = []

list_circular_correlation_averted_baseline_pre_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_baseline_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_averted = (
            path_2_baseline_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_averted_baseline_pre_suffix
        )
        fname2_averted = (
            path_2_baseline_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_averted_baseline_pre_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 10:
            fname2_averted = (
                path_2_baseline_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_baseline_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_baseline_pre_suffix
        )
        fname2_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_baseline_pre_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 20:
            fname2_averted = (
                path_2_baseline_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_baseline_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_baseline_pre_suffix
        )
        fname2_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_baseline_pre_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 30:
            fname2_averted = (
                path_2_baseline_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_baseline_pre_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_averted = fname1_averted
    fname_S2_averted = fname2_averted

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_averted = mne.io.read_raw_fif(fname_S1_averted, preload=True, verbose=False)
    raw2_averted = mne.io.read_raw_fif(fname_S2_averted, preload=True, verbose=False)

    raw1_averted.info["bads"] = original_bad_channels1
    raw2_averted.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_averted = raw1_averted.filter(l_freq=1, h_freq=40)
    raw2_averted = raw2_averted.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_averted = raw1_averted.copy().interpolate_bads(reset_bads=True)
    raw2_averted = raw2_averted.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_averted, id=1, duration=1)

    # Epoch length is 1 second
    epo1_averted = mne.Epochs(
        raw1_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_averted = mne.Epochs(
        raw2_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_averted, epo2_averted])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_averted.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_averted, epo2_averted],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_averted, epo2_averted], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_averted_baseline_pre.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_averted_baseline_pre_no_filter_all.append(theta)
    list_circular_correlation_averted_baseline_pre_no_filter_all.append(alpha)
    list_circular_correlation_averted_baseline_pre_no_filter_all.append(beta)
    list_circular_correlation_averted_baseline_pre_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-averted) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    preproc_filename1 = fname_S1_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_averted.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_averted.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)

# Change to a avertedory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_averted_baseline_pre_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_averted_baseline_pre_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-averted) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_averted_baseline_pre.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_averted_baseline_pre, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
print("(pre-averted) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% [markdown]
# ## Baseline averted post

# %%
# Container for no. of connections of ALL participants
path_2_baseline_data_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
path_2_dir_2_save_preprocessed_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/"
path_2_dir_2_save_raw_preprocessed_epoched_data = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_epoched_data/"
odd_subject_averted_baseline_post_suffix = "-averted_pre_left_right_point_combined_raw.fif"
even_subject_averted_baseline_post_suffix = "-averted_pre_right_left_point_combined_raw.fif"


start = timer()

all_deleted_epochs_indices_averted_baseline_post = []

total_n_connections_all_pairs_averted_baseline_post = []

list_circular_correlation_scores_all_theta = []
list_circular_correlation_scores_all_alpha = []
list_circular_correlation_scores_all_beta = []
list_circular_correlation_scores_all_gamma = []
list_circular_correlation_scores_all_averted_baseline_post = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_averted_baseline_post = []

list_circular_correlation_averted_baseline_post_no_filter_all = []

# TODO :Define bad channels for each participant
original_bad_channels_all = [
    ["FP1", "C3", "T7"],
    ["FP1", "F7", "C4"],
    ["FP1", "Fp2", "F7", "C4"],
    ["FP1"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    ["FP1", "Fp2", "F7", "C4", "P4"],
    [],
    ["Fp2", "C3"],
    ["F3"],
    ["Fp2", "F4", "C3", "P3"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
    ["FP1", "T7", "C3", "P4"],
    [],
    ["Fp2", "C3", "P3"],
    [],
    ["Fp2", "F3", "C3"],
    ["F7", "F3", "T7", "P8"],
    ["Fp2", "C3", "P3", "P4", "O1"],
    [],
    ["Fp2", "C3"],
    ["P7"],
    ["Fp2", "C3", "O1"],
    ["P3", "P4"],
    ["Fp2", "C3", "P4"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["Fp2", "C3"],
    [],
    ["Fp2", "C3"],
    ["P3", "P4"],
    ["FP1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3", "C4"],
]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)

# TODO : Adjust the loop number. Now it is only up to 16 files (so far)
begin = 0
end = 16
step = 2

for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    os.chdir(path_2_baseline_data_dir)

    # If we want to exclude pair that wants to be skipped
    # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
    # Uncomment this line, in case you have files that want to be skipped
    # if (i == 2):  # NOTE: Indicate pair
    #     continue

    # Subject no. 1 - 10
    if (i + 1) <= 9:
        fname1_averted = (
            path_2_baseline_data_dir
            + "S0"
            + str(i + 1)
            + odd_subject_averted_baseline_post_suffix
        )
        fname2_averted = (
            path_2_baseline_data_dir
            + "S0"
            + str(i + 2)
            + even_subject_averted_baseline_post_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 10:
            fname2_averted = (
                path_2_baseline_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_baseline_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 11 - 20
    elif (i + 1) >= 11:
        fname1_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_baseline_post_suffix
        )
        fname2_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_baseline_post_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 20:
            fname2_averted = (
                path_2_baseline_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_baseline_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    # Subject no. 21 - 30
    elif (i + 1) >= 21:
        fname1_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 1)
            + odd_subject_averted_baseline_post_suffix
        )
        fname2_averted = (
            path_2_baseline_data_dir
            + "S"
            + str(i + 2)
            + even_subject_averted_baseline_post_suffix
        )
        # Replace fname2_averted variable
        if (i + 2) == 30:
            fname2_averted = (
                path_2_baseline_data_dir
                + "S"
                + str(i + 2)
                + even_subject_averted_baseline_post_suffix
            )

        # Indicator of which files are being processed
        print(f"Processing S-{i + 1} & S-{i + 2}")

    freq_bands = {
        "Theta": [4, 7],
        "Alpha": [7.5, 13],
        "Beta": [13.5, 29.5],
        "Gamma": [30, 40],
    }

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_averted = fname1_averted
    fname_S2_averted = fname2_averted

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]

    raw1_averted = mne.io.read_raw_fif(fname_S1_averted, preload=True, verbose=False)
    raw2_averted = mne.io.read_raw_fif(fname_S2_averted, preload=True, verbose=False)

    raw1_averted.info["bads"] = original_bad_channels1
    raw2_averted.info["bads"] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_averted = raw1_averted.filter(l_freq=1, h_freq=40)
    raw2_averted = raw2_averted.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_averted = raw1_averted.copy().interpolate_bads(reset_bads=True)
    raw2_averted = raw2_averted.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_averted, id=1, duration=1)

    # Epoch length is 1 second
    epo1_averted = mne.Epochs(
        raw1_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epo2_averted = mne.Epochs(
        raw2_averted,
        events,
        tmin=0.0,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    mne.epochs.equalize_epoch_counts([epo1_averted, epo2_averted])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_averted.info["sfreq"]
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit(
        [epo1_averted, epo2_averted],
        n_components=15,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
    )

    cleaned_epochs_ICA = prep.ICA_autocorrect(
        icas, [epo1_averted, epo2_averted], verbose=True
    )

    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # Populate indices of bad epochs into a list. Now are there are 3 outputs
    cleaned_epochs_AR, percentage_rejected_epoch, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=True
    )

    # Populate indices of deleted epochs into a list. We need this to reject epochs in eye tracker data
    # length of the list will be a half of number of participants
    all_deleted_epochs_indices_averted_baseline_post.append(delete_epochs_indices)

    # Load and Picking the preprocessed epochs for each participant
    # With ICA
    preproc_S1 = cleaned_epochs_AR[0]
    preproc_S2 = cleaned_epochs_AR[1]

    #
    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        preproc_S1, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    psd2 = analyses.pow(
        preproc_S2, fmin=4, fmax=40, n_fft=1000, n_per_seg=1000, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # # Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
    result = analyses.compute_sync(complex_signal, mode="ccorr")
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info["ch_names"])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

    list_circular_correlation_averted_baseline_post_no_filter_all.append(theta)
    list_circular_correlation_averted_baseline_post_no_filter_all.append(alpha)
    list_circular_correlation_averted_baseline_post_no_filter_all.append(beta)
    list_circular_correlation_averted_baseline_post_no_filter_all.append(gamma)

    # Check if inter-brain connection scores have been put into a list
    print(
        f"(pre-averted) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
        have been put into a list (theta, alpha, beta, gamma)"
    )

    preproc_filename1 = fname_S1_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_1 = fname_S1_averted.find("S") 
    epoched_file_name_S1 = preproc_filename1[s_indicator_index_1:].split(" "[0])
    epoched_file_name_S1 = epoched_file_name_S1[0]
    epoched_file_name_S1 = epoched_file_name_S1[:-4] + "-epo.fif"

    # Extract file name from path (subject 2)
    preproc_filename2 = fname_S2_averted
    # NOTE Be careful when there is "S" in path
    s_indicator_index_2 = fname_S2_averted.find("S") 
    epoched_file_name_S2 = preproc_filename2[s_indicator_index_2:].split(" "[0])
    epoched_file_name_S2 = epoched_file_name_S2[0]
    epoched_file_name_S2 = epoched_file_name_S2[:-4] + "-epo.fif"

    # Change to a directory where we want to save raw pre processed epoched data
    os.chdir(path_2_dir_2_save_raw_preprocessed_epoched_data)

    # Save pre-processed (epoched) data of subject 1
    preproc_S1.save(epoched_file_name_S1, overwrite=True)

    # Save pre-processed (epoched) data of subject 2
    preproc_S2.save(epoched_file_name_S2, overwrite=True)

# Change to a avertedory where we want to save the above populated lists (pre-processed data)
os.chdir(path_2_dir_2_save_preprocessed_data)

# Save the scores of inter-brain synchrony from each pair into pkl file
with open(
    "list_circular_correlation_scores_all_pairs_averted_baseline_post_no_filter.pkl", "wb"
) as handle:
    pickle.dump(
        list_circular_correlation_averted_baseline_post_no_filter_all,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

# NOTE : The structure of files is each pair will have 4 lists, which has the following order
#        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to subject 1,
#        * then move the 2nd four lists which belong to subject 2 and so on.
print(
    "(pre-averted) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
)

# Save indices of deleted epochs from each pair into pkl file
# NOTE : Length of list once pkl file is loaded is equal to the number of pairs
# If we have 15 pairs, then there will be 15 lists within that pkl file
with open("list_deleted_epoch_indices_averted_baseline_post.pkl", "wb") as handle:
    pickle.dump(
        all_deleted_epochs_indices_averted_baseline_post, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
print("(pre-averted) All indices of deleted epochs have been saved into a pickle file")


# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% [markdown]
# ## Add : Read pkl file
# * Populate by frequency (theta, alpha, beta, and gamma)

# %%
# TODO: Save list of power-correlation scores into pickle file
# os.getcwd()

# Read power-correlation scores list
# with open('list_circular_correlation_scores_all_pairs_direct_pre_no_filter.pkl', 'rb') as handle:
#     circular_correlation_theta = pickle.load(handle)
#     circular_correlation_theta

#
# # TODO: GO HERE !! Average real power-correlation scores for each pair. Total there are 15 lists
# avg_circular_correlation_theta = []
# for i, val in enumerate(circular_correlation_theta):
#     if val == []:
#         avg_circular_correlation_theta.append(0)
#         continue
#     avg_circular_correlation_theta.append(statistics.mean(val))
