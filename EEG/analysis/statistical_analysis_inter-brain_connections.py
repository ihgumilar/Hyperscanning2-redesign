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
#     display_name: Python 3.8.10 ('hyperscanning2_redesign_new')
#     language: python
#     name: python3
# ---

# %%
### Relevant packages
# If there is an error in importing LabelConter, then run the following line to change working directory
# Then run the above cell again

import os

os.chdir("/hpc/igum002/codes/Hyperscanning2-redesign/EEG/analysis")
import pickle
from collections import Counter, OrderedDict

import mne
import numpy as np
import pandas as pd
import scipy.stats as stats
from hypyp import analyses, prep, stats, viz
from tqdm import tqdm

from LabelConverter import get_electrode_labels_connections

# %% [markdown]
# #### Just in case failed in importing LabelConverter

# %%
# If there is an error in importing LabelConter, then run the following line to change working directory
# Then run the above cell again
os.chdir("/hpc/igum002/codes/Hyperscanning2-redesign/EEG/analysis")

# %% [markdown]
# ### Statistical analysis (averted_pre)

# %%

# TODO Where preprocessed files are stored (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
path_2_preproc_averted_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/averted_pre/"

# TODO Directory to save significant connections (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
saved_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre/"

# IMPORTANT ! how many permutations you want
n_perms = 150

# List files that are contained in a specified directory
list_of_files = os.listdir(path_2_preproc_averted_pre)

# Change working directory
os.chdir(path_2_preproc_averted_pre)

# To loop subject number.
begin = 0
end = len(list_of_files)
step = 2

freq_bands = {
    "Theta": [4, 7],
    "Alpha": [7.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 40],
}
freq_bands = OrderedDict(freq_bands)

ch_names = [
    "FP1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
ch_types = ["eeg"] * 16
info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)


for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    filename1 = list_of_files[i]
    filename2 = list_of_files[i + 1]

    # Load preprocessed epochs
    load_epoch_S1 = mne.read_epochs(filename1, preload=True)
    load_epoch_S2 = mne.read_epochs(filename2, preload=True)

    # Equalize number of epochs
    mne.epochs.equalize_epoch_counts([load_epoch_S1, load_epoch_S2])

    sampling_rate = load_epoch_S1.info["sfreq"]

    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        load_epoch_S1, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    psd2 = analyses.pow(
        load_epoch_S2, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    # ### Connectivity
    # with ICA
    data_inter = np.array([load_epoch_S1, load_epoch_S2])
    result_intra = []

    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

    # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
    ground_result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
    ground_result_plv = analyses.compute_sync(complex_signal, mode="plv")
    ground_result_coh = analyses.compute_sync(complex_signal, mode="coh")

    # Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    n_ch = len(load_epoch_S1.info["ch_names"])
    theta_ccorr, alpha_ccorr, beta_ccorr, gamma_ccorr = ground_result_ccorr[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_plv, alpha_plv, beta_plv, gamma_plv = ground_result_plv[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_coh, alpha_coh, beta_coh, gamma_coh = ground_result_coh[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]

    # TODO Get the type of data this and change the type to ndarray in later down the section
    # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
    # ground truth matrix using ccorr
    ccorr_combined_ground_truth_matrices = [
        theta_ccorr,
        alpha_ccorr,
        beta_ccorr,
        gamma_ccorr,
    ]
    # ground truth matrix using plv
    plv_combined_ground_truth_matrices = [theta_plv, alpha_plv, beta_plv, gamma_plv]
    # ground truth matrix using coh
    coh_combined_ground_truth_matrices = [theta_coh, alpha_coh, beta_coh, gamma_coh]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for ccorr)
    ccorr_theta_n_connections = np.zeros([16, 16])
    ccorr_alpha_n_connections = np.zeros([16, 16])
    ccorr_beta_n_connections = np.zeros([16, 16])
    ccorr_gamma_n_connections = np.zeros([16, 16])
    ccorr_combined_freq_n_connections = [
        ccorr_theta_n_connections,
        ccorr_alpha_n_connections,
        ccorr_beta_n_connections,
        ccorr_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for plv)
    plv_theta_n_connections = np.zeros([16, 16])
    plv_alpha_n_connections = np.zeros([16, 16])
    plv_beta_n_connections = np.zeros([16, 16])
    plv_gamma_n_connections = np.zeros([16, 16])
    plv_combined_freq_n_connections = [
        plv_theta_n_connections,
        plv_alpha_n_connections,
        plv_beta_n_connections,
        plv_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for coh)
    coh_theta_n_connections = np.zeros([16, 16])
    coh_alpha_n_connections = np.zeros([16, 16])
    coh_beta_n_connections = np.zeros([16, 16])
    coh_gamma_n_connections = np.zeros([16, 16])
    coh_combined_freq_n_connections = [
        coh_theta_n_connections,
        coh_alpha_n_connections,
        coh_beta_n_connections,
        coh_gamma_n_connections,
    ]

    ############ Implemanting Permutation test ################################

    # TODO Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

    ccorr_combined_freqs_electrode_pair_n_actual_score = []
    plv_combined_freqs_electrode_pair_n_actual_score = []
    coh_combined_freqs_electrode_pair_n_actual_score = []

    electrode_pair_n_actual_score_theta_ccorr = []
    electrode_pair_n_actual_score_alpha_ccorr = []
    electrode_pair_n_actual_score_beta_ccorr = []
    electrode_pair_n_actual_score_gamma_ccorr = []

    electrode_pair_n_actual_score_theta_plv = []
    electrode_pair_n_actual_score_alpha_plv = []
    electrode_pair_n_actual_score_beta_plv = []
    electrode_pair_n_actual_score_gamma_plv = []

    electrode_pair_n_actual_score_theta_coh = []
    electrode_pair_n_actual_score_alpha_coh = []
    electrode_pair_n_actual_score_beta_coh = []
    electrode_pair_n_actual_score_gamma_coh = []

    for participant1_channel in range(len(ch_names)):
        for participant2_channel in range(len(ch_names)):

            # epoch1 should just be a specific electrode e.g. FP1.
            # epoch1 = load_epoch_S1.pick_channels('FP1') or something. <- note this is not correct, it's just an idea
            epoch1 = load_epoch_S1.get_data(picks=ch_names[participant1_channel])
            epoch2 = load_epoch_S2.get_data(picks=ch_names[participant2_channel])

            rng = np.random.default_rng(42)  # set a random seed

            # Permute for several times as defined above
            # TODO Change this to 80 for real one
            n_perms = 150

            k_ccorr_theta_permuted = (
                []
            )  # initialising list that will store all the ccor values (ccorr value range of 0 to +1). its length should equal n_perms
            k_ccorr_alpha_permuted = []
            k_ccorr_beta_permuted = []
            k_ccorr_gamma_permuted = []

            k_plv_theta_permuted = (
                []
            )  # initialising list that will store all the plv values (plv value range of 0 to +1). its length should equal n_perms
            k_plv_alpha_permuted = []
            k_plv_beta_permuted = []
            k_plv_gamma_permuted = []

            k_coh_theta_permuted = (
                []
            )  # initialising list that will store all the coh values (coh value range of 0 to +1). its length should equal n_perms
            k_coh_alpha_permuted = []
            k_coh_beta_permuted = []
            k_coh_gamma_permuted = []

            # for each iterations, undergo permutation and calculate ccorr or plv or coh
            for iperm in range(n_perms):

                # for participant 1
                perm1 = rng.permutation(
                    len(epoch1)
                )  # randomising indices without replacement
                epoch1_in_permuted_order = [
                    epoch1[i] for i in perm1
                ]  # index the epoch to get permuted epochs

                # for participant 2
                perm2 = rng.permutation(len(epoch2))
                epoch2_in_permuted_order = [epoch1[i] for i in perm2]

                # combine the two permuted epochs together
                data_inter_permuted = np.array([epoch1, epoch2])

                # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                complex_signal = analyses.compute_freq_bands(
                    data_inter_permuted, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
                result_plv = analyses.compute_sync(complex_signal, mode="plv")
                result_coh = analyses.compute_sync(complex_signal, mode="coh")

                # comparing 1 channel of participant 1 against 1 channel of participant 2
                # n_ch is 1 instead of 16 because we are finding connection of 1 electrode and 1 other electrode each time.
                n_ch = 1
                # slice the result array to seperate into different frequencies. do this for 'ccorr', 'plv' and 'coh'
                (
                    ccorr_theta_permuted,
                    ccorr_alpha_permuted,
                    ccorr_beta_permuted,
                    ccorr_gamma_permuted,
                ) = result_ccorr[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    plv_theta_permuted,
                    plv_alpha_permuted,
                    plv_beta_permuted,
                    plv_gamma_permuted,
                ) = result_plv[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    coh_theta_permuted,
                    coh_alpha_permuted,
                    coh_beta_permuted,
                    coh_gamma_permuted,
                ) = result_coh[:, 0:n_ch, n_ch : 2 * n_ch]

                # append the ccorr value to the corresponding list
                k_ccorr_theta_permuted.append(ccorr_theta_permuted)
                k_ccorr_alpha_permuted.append(ccorr_alpha_permuted)
                k_ccorr_beta_permuted.append(ccorr_beta_permuted)
                k_ccorr_gamma_permuted.append(ccorr_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta ccorr permutation scores, 2nd index stores alpha ccorr permutation scores, etc.
                combined_k_ccorr_frequency_permuted = [
                    k_ccorr_theta_permuted,
                    k_ccorr_alpha_permuted,
                    k_ccorr_beta_permuted,
                    k_ccorr_gamma_permuted,
                ]

                # append the plv value to the corresponding list
                k_plv_theta_permuted.append(plv_theta_permuted)
                k_plv_alpha_permuted.append(plv_alpha_permuted)
                k_plv_beta_permuted.append(plv_beta_permuted)
                k_plv_gamma_permuted.append(plv_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta plv permutation scores, 2nd index stores alpha plv permutation scores, etc.
                combined_k_plv_frequency_permuted = [
                    k_plv_theta_permuted,
                    k_plv_alpha_permuted,
                    k_plv_beta_permuted,
                    k_plv_gamma_permuted,
                ]

                # append the coh value to the corresponding list
                k_coh_theta_permuted.append(coh_theta_permuted)
                k_coh_alpha_permuted.append(coh_alpha_permuted)
                k_coh_beta_permuted.append(coh_beta_permuted)
                k_coh_gamma_permuted.append(coh_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta coh permutation scores, 2nd index stores alpha coh permutation scores, etc.
                combined_k_coh_frequency_permuted = [
                    k_coh_theta_permuted,
                    k_coh_alpha_permuted,
                    k_coh_beta_permuted,
                    k_coh_gamma_permuted,
                ]

            # iterate each theta, alpha, beta, gamma
            for iterate_each_freq in range(
                len(combined_k_ccorr_frequency_permuted)
            ):  # 4 because we have classified frequency range into 4 types: theta, alpha, beta, gamma

                # Calculate p value
                z_value = 1.96  # equivalent to p value of 0.05

                # calculate mean and standard deviation for each frequency band using ccorr
                ccorr_mean_permuted = np.mean(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_std_permuted = np.std(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_z_score = (
                    ccorr_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - ccorr_mean_permuted
                ) / ccorr_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(ccorr_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    ccorr_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using plv
                plv_mean_permuted = np.mean(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_std_permuted = np.std(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_z_score = (
                    plv_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - plv_mean_permuted
                ) / plv_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(plv_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    plv_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using coh
                coh_mean_permuted = np.mean(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_std_permuted = np.std(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_z_score = (
                    coh_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - coh_mean_permuted
                ) / coh_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(coh_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    coh_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

    # convert the 4 x 16 x 16 array into a list (marker significant connection matrix)
    ccorr_combined_freq_n_connections_list = list(ccorr_combined_freq_n_connections)
    plv_combined_freq_n_connections_list = list(plv_combined_freq_n_connections)
    coh_combined_freq_n_connections_list = list(coh_combined_freq_n_connections)

    # Progress getting actual score from marker significant connection matrix (ccorr)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_ccorr = []
    alpha_sig_electrode_pair_ccorr = []
    beta_sig_electrode_pair_ccorr = []
    gamma_sig_electrode_pair_ccorr = []

    for idx_freq in range(len(ccorr_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(ccorr_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                ccorr_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if ccorr_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = ccorr_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_ccorr = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for ccorr)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )

    # Create main list that contains all the above 4 lists of frequency.
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        beta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_ccorr
    )

    # Progress getting actual score from marker significant connection matrix (plv)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_plv = []
    alpha_sig_electrode_pair_plv = []
    beta_sig_electrode_pair_plv = []
    gamma_sig_electrode_pair_plv = []

    for idx_freq in range(len(plv_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(plv_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                plv_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if plv_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = plv_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_plv = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for plv)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )

    # Create main list that contains all the above 4 lists of frequency.
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_plv)
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_plv
    )

    # Progress getting actual score from marker significant connection matrix (coh)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_coh = []
    alpha_sig_electrode_pair_coh = []
    beta_sig_electrode_pair_coh = []
    gamma_sig_electrode_pair_coh = []

    for idx_freq in range(len(coh_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(coh_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                coh_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if coh_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = coh_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_coh = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for coh)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )

    # Create main list that contains all the above 4 lists of frequency.
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_coh)
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_coh
    )

    # So there will be 3 main containers (actual score of ccor, plv, coh). Each of them has 4 lists (theta, alpha, beta, and gamma)

    # save ccorr connection data for a pair
    saved_filename1 = (
        saved_directory
        + "Pre_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of ccorr for a pair
    saved_actual_score_filename1 = (
        saved_directory
        + "Pre_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save plv connection data for a pair
    saved_filename2 = (
        saved_directory
        + "Pre_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of plv for a pair
    saved_actual_score_filename2 = (
        saved_directory
        + "Pre_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save coh connection data for a pair
    saved_filename3 = (
        saved_directory
        + "Pre_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of coh for a pair
    saved_actual_score_filename3 = (
        saved_directory
        + "Pre_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# %% [markdown]
# ### Statistical analysis (averted_post)

# %%
# TODO Where preprocessed files are stored (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
path_2_preproc_averted_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/averted_post/"

# TODO Directory to save significant connections (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
saved_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_post/"

# IMPORTANT ! how many permutations you want
n_perms = 150

# List files that are contained in a specified directory
list_of_files = os.listdir(path_2_preproc_averted_post)

# Change working directory
os.chdir(path_2_preproc_averted_post)

# To loop subject number.
begin = 0
end = len(list_of_files)
step = 2

freq_bands = {
    "Theta": [4, 7],
    "Alpha": [7.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 40],
}
freq_bands = OrderedDict(freq_bands)

ch_names = [
    "FP1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
ch_types = ["eeg"] * 16
info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)


for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    filename1 = list_of_files[i]
    filename2 = list_of_files[i + 1]

    # Load preprocessed epochs
    load_epoch_S1 = mne.read_epochs(filename1, preload=True)
    load_epoch_S2 = mne.read_epochs(filename2, preload=True)

    # Equalize number of epochs
    mne.epochs.equalize_epoch_counts([load_epoch_S1, load_epoch_S2])

    sampling_rate = load_epoch_S1.info["sfreq"]

    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        load_epoch_S1, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    psd2 = analyses.pow(
        load_epoch_S2, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    # ### Connectivity
    # with ICA
    data_inter = np.array([load_epoch_S1, load_epoch_S2])
    result_intra = []

    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

    # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
    ground_result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
    ground_result_plv = analyses.compute_sync(complex_signal, mode="plv")
    ground_result_coh = analyses.compute_sync(complex_signal, mode="coh")

    # Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    n_ch = len(load_epoch_S1.info["ch_names"])
    theta_ccorr, alpha_ccorr, beta_ccorr, gamma_ccorr = ground_result_ccorr[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_plv, alpha_plv, beta_plv, gamma_plv = ground_result_plv[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_coh, alpha_coh, beta_coh, gamma_coh = ground_result_coh[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]

    # TODO Get the type of data this and change the type to ndarray in later down the section
    # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
    # ground truth matrix using ccorr
    ccorr_combined_ground_truth_matrices = [
        theta_ccorr,
        alpha_ccorr,
        beta_ccorr,
        gamma_ccorr,
    ]
    # ground truth matrix using plv
    plv_combined_ground_truth_matrices = [theta_plv, alpha_plv, beta_plv, gamma_plv]
    # ground truth matrix using coh
    coh_combined_ground_truth_matrices = [theta_coh, alpha_coh, beta_coh, gamma_coh]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for ccorr)
    ccorr_theta_n_connections = np.zeros([16, 16])
    ccorr_alpha_n_connections = np.zeros([16, 16])
    ccorr_beta_n_connections = np.zeros([16, 16])
    ccorr_gamma_n_connections = np.zeros([16, 16])
    ccorr_combined_freq_n_connections = [
        ccorr_theta_n_connections,
        ccorr_alpha_n_connections,
        ccorr_beta_n_connections,
        ccorr_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for plv)
    plv_theta_n_connections = np.zeros([16, 16])
    plv_alpha_n_connections = np.zeros([16, 16])
    plv_beta_n_connections = np.zeros([16, 16])
    plv_gamma_n_connections = np.zeros([16, 16])
    plv_combined_freq_n_connections = [
        plv_theta_n_connections,
        plv_alpha_n_connections,
        plv_beta_n_connections,
        plv_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for coh)
    coh_theta_n_connections = np.zeros([16, 16])
    coh_alpha_n_connections = np.zeros([16, 16])
    coh_beta_n_connections = np.zeros([16, 16])
    coh_gamma_n_connections = np.zeros([16, 16])
    coh_combined_freq_n_connections = [
        coh_theta_n_connections,
        coh_alpha_n_connections,
        coh_beta_n_connections,
        coh_gamma_n_connections,
    ]

    ############ Implemanting Permutation test ################################

    # TODO Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

    ccorr_combined_freqs_electrode_pair_n_actual_score = []
    plv_combined_freqs_electrode_pair_n_actual_score = []
    coh_combined_freqs_electrode_pair_n_actual_score = []

    electrode_pair_n_actual_score_theta_ccorr = []
    electrode_pair_n_actual_score_alpha_ccorr = []
    electrode_pair_n_actual_score_beta_ccorr = []
    electrode_pair_n_actual_score_gamma_ccorr = []

    electrode_pair_n_actual_score_theta_plv = []
    electrode_pair_n_actual_score_alpha_plv = []
    electrode_pair_n_actual_score_beta_plv = []
    electrode_pair_n_actual_score_gamma_plv = []

    electrode_pair_n_actual_score_theta_coh = []
    electrode_pair_n_actual_score_alpha_coh = []
    electrode_pair_n_actual_score_beta_coh = []
    electrode_pair_n_actual_score_gamma_coh = []

    for participant1_channel in range(len(ch_names)):
        for participant2_channel in range(len(ch_names)):

            # epoch1 should just be a specific electrode e.g. FP1.
            # epoch1 = load_epoch_S1.pick_channels('FP1') or something. <- note this is not correct, it's just an idea
            epoch1 = load_epoch_S1.get_data(picks=ch_names[participant1_channel])
            epoch2 = load_epoch_S2.get_data(picks=ch_names[participant2_channel])

            rng = np.random.default_rng(42)  # set a random seed

            # Permute for several times as defined above
            # TODO Change this to 80 for real one
            n_perms = 150

            k_ccorr_theta_permuted = (
                []
            )  # initialising list that will store all the ccor values (ccorr value range of 0 to +1). its length should equal n_perms
            k_ccorr_alpha_permuted = []
            k_ccorr_beta_permuted = []
            k_ccorr_gamma_permuted = []

            k_plv_theta_permuted = (
                []
            )  # initialising list that will store all the plv values (plv value range of 0 to +1). its length should equal n_perms
            k_plv_alpha_permuted = []
            k_plv_beta_permuted = []
            k_plv_gamma_permuted = []

            k_coh_theta_permuted = (
                []
            )  # initialising list that will store all the coh values (coh value range of 0 to +1). its length should equal n_perms
            k_coh_alpha_permuted = []
            k_coh_beta_permuted = []
            k_coh_gamma_permuted = []

            # for each iterations, undergo permutation and calculate ccorr or plv or coh
            for iperm in range(n_perms):

                # for participant 1
                perm1 = rng.permutation(
                    len(epoch1)
                )  # randomising indices without replacement
                epoch1_in_permuted_order = [
                    epoch1[i] for i in perm1
                ]  # index the epoch to get permuted epochs

                # for participant 2
                perm2 = rng.permutation(len(epoch2))
                epoch2_in_permuted_order = [epoch1[i] for i in perm2]

                # combine the two permuted epochs together
                data_inter_permuted = np.array([epoch1, epoch2])

                # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                complex_signal = analyses.compute_freq_bands(
                    data_inter_permuted, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
                result_plv = analyses.compute_sync(complex_signal, mode="plv")
                result_coh = analyses.compute_sync(complex_signal, mode="coh")

                # comparing 1 channel of participant 1 against 1 channel of participant 2
                # n_ch is 1 instead of 16 because we are finding connection of 1 electrode and 1 other electrode each time.
                n_ch = 1
                # slice the result array to seperate into different frequencies. do this for 'ccorr', 'plv' and 'coh'
                (
                    ccorr_theta_permuted,
                    ccorr_alpha_permuted,
                    ccorr_beta_permuted,
                    ccorr_gamma_permuted,
                ) = result_ccorr[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    plv_theta_permuted,
                    plv_alpha_permuted,
                    plv_beta_permuted,
                    plv_gamma_permuted,
                ) = result_plv[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    coh_theta_permuted,
                    coh_alpha_permuted,
                    coh_beta_permuted,
                    coh_gamma_permuted,
                ) = result_coh[:, 0:n_ch, n_ch : 2 * n_ch]

                # append the ccorr value to the corresponding list
                k_ccorr_theta_permuted.append(ccorr_theta_permuted)
                k_ccorr_alpha_permuted.append(ccorr_alpha_permuted)
                k_ccorr_beta_permuted.append(ccorr_beta_permuted)
                k_ccorr_gamma_permuted.append(ccorr_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta ccorr permutation scores, 2nd index stores alpha ccorr permutation scores, etc.
                combined_k_ccorr_frequency_permuted = [
                    k_ccorr_theta_permuted,
                    k_ccorr_alpha_permuted,
                    k_ccorr_beta_permuted,
                    k_ccorr_gamma_permuted,
                ]

                # append the plv value to the corresponding list
                k_plv_theta_permuted.append(plv_theta_permuted)
                k_plv_alpha_permuted.append(plv_alpha_permuted)
                k_plv_beta_permuted.append(plv_beta_permuted)
                k_plv_gamma_permuted.append(plv_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta plv permutation scores, 2nd index stores alpha plv permutation scores, etc.
                combined_k_plv_frequency_permuted = [
                    k_plv_theta_permuted,
                    k_plv_alpha_permuted,
                    k_plv_beta_permuted,
                    k_plv_gamma_permuted,
                ]

                # append the coh value to the corresponding list
                k_coh_theta_permuted.append(coh_theta_permuted)
                k_coh_alpha_permuted.append(coh_alpha_permuted)
                k_coh_beta_permuted.append(coh_beta_permuted)
                k_coh_gamma_permuted.append(coh_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta coh permutation scores, 2nd index stores alpha coh permutation scores, etc.
                combined_k_coh_frequency_permuted = [
                    k_coh_theta_permuted,
                    k_coh_alpha_permuted,
                    k_coh_beta_permuted,
                    k_coh_gamma_permuted,
                ]

            # iterate each theta, alpha, beta, gamma
            for iterate_each_freq in range(
                len(combined_k_ccorr_frequency_permuted)
            ):  # 4 because we have classified frequency range into 4 types: theta, alpha, beta, gamma

                # Calculate p value
                z_value = 1.96  # equivalent to p value of 0.05

                # calculate mean and standard deviation for each frequency band using ccorr
                ccorr_mean_permuted = np.mean(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_std_permuted = np.std(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_z_score = (
                    ccorr_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - ccorr_mean_permuted
                ) / ccorr_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(ccorr_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    ccorr_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using plv
                plv_mean_permuted = np.mean(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_std_permuted = np.std(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_z_score = (
                    plv_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - plv_mean_permuted
                ) / plv_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(plv_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    plv_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using coh
                coh_mean_permuted = np.mean(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_std_permuted = np.std(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_z_score = (
                    coh_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - coh_mean_permuted
                ) / coh_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(coh_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    coh_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

    # convert the 4 x 16 x 16 array into a list (marker significant connection matrix)
    ccorr_combined_freq_n_connections_list = list(ccorr_combined_freq_n_connections)
    plv_combined_freq_n_connections_list = list(plv_combined_freq_n_connections)
    coh_combined_freq_n_connections_list = list(coh_combined_freq_n_connections)

    # Progress getting actual score from marker significant connection matrix (ccorr)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_ccorr = []
    alpha_sig_electrode_pair_ccorr = []
    beta_sig_electrode_pair_ccorr = []
    gamma_sig_electrode_pair_ccorr = []

    for idx_freq in range(len(ccorr_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(ccorr_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                ccorr_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if ccorr_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = ccorr_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_ccorr = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for ccorr)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )

    # Create main list that contains all the above 4 lists of frequency.
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        beta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_ccorr
    )

    # Progress getting actual score from marker significant connection matrix (plv)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_plv = []
    alpha_sig_electrode_pair_plv = []
    beta_sig_electrode_pair_plv = []
    gamma_sig_electrode_pair_plv = []

    for idx_freq in range(len(plv_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(plv_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                plv_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if plv_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = plv_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_plv = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for plv)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )

    # Create main list that contains all the above 4 lists of frequency.
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_plv)
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_plv
    )

    # Progress getting actual score from marker significant connection matrix (coh)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_coh = []
    alpha_sig_electrode_pair_coh = []
    beta_sig_electrode_pair_coh = []
    gamma_sig_electrode_pair_coh = []

    for idx_freq in range(len(coh_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(coh_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                coh_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if coh_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = coh_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_coh = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for coh)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )

    # Create main list that contains all the above 4 lists of frequency.
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_coh)
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_coh
    )

    # So there will be 3 main containers (actual score of ccor, plv, coh). Each of them has 4 lists (theta, alpha, beta, and gamma)

    # save ccorr connection data for a pair
    saved_filename1 = (
        saved_directory
        + "Post_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of ccorr for a pair
    saved_actual_score_filename1 = (
        saved_directory
        + "Post_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save plv connection data for a pair
    saved_filename2 = (
        saved_directory
        + "Post_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of plv for a pair
    saved_actual_score_filename2 = (
        saved_directory
        + "Post_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save coh connection data for a pair
    saved_filename3 = (
        saved_directory
        + "Post_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of coh for a pair
    saved_actual_score_filename3 = (
        saved_directory
        + "Post_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

# %% [markdown]
# ### Statistical analysis (direct_pre)

# %%

# TODO Where preprocessed files are stored (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
path_2_preproc_direct_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/direct_pre/"

# TODO Directory to save significant connections (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
saved_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_pre/"

# IMPORTANT ! how many permutations you want
n_perms = 150

# List files that are contained in a specified directory
list_of_files = os.listdir(path_2_preproc_direct_pre)

# Change working directory
os.chdir(path_2_preproc_direct_pre)

# To loop subject number.
begin = 0
end = len(list_of_files)
step = 2

freq_bands = {
    "Theta": [4, 7],
    "Alpha": [7.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 40],
}
freq_bands = OrderedDict(freq_bands)

ch_names = [
    "FP1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
ch_types = ["eeg"] * 16
info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)


for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    filename1 = list_of_files[i]
    filename2 = list_of_files[i + 1]

    # Load preprocessed epochs
    load_epoch_S1 = mne.read_epochs(filename1, preload=True)
    load_epoch_S2 = mne.read_epochs(filename2, preload=True)

    # Equalize number of epochs
    mne.epochs.equalize_epoch_counts([load_epoch_S1, load_epoch_S2])

    sampling_rate = load_epoch_S1.info["sfreq"]

    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        load_epoch_S1, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    psd2 = analyses.pow(
        load_epoch_S2, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    # ### Connectivity
    # with ICA
    data_inter = np.array([load_epoch_S1, load_epoch_S2])
    result_intra = []

    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

    # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
    ground_result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
    ground_result_plv = analyses.compute_sync(complex_signal, mode="plv")
    ground_result_coh = analyses.compute_sync(complex_signal, mode="coh")

    # Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    n_ch = len(load_epoch_S1.info["ch_names"])
    theta_ccorr, alpha_ccorr, beta_ccorr, gamma_ccorr = ground_result_ccorr[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_plv, alpha_plv, beta_plv, gamma_plv = ground_result_plv[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_coh, alpha_coh, beta_coh, gamma_coh = ground_result_coh[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]

    # TODO Get the type of data this and change the type to ndarray in later down the section
    # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
    # ground truth matrix using ccorr
    ccorr_combined_ground_truth_matrices = [
        theta_ccorr,
        alpha_ccorr,
        beta_ccorr,
        gamma_ccorr,
    ]
    # ground truth matrix using plv
    plv_combined_ground_truth_matrices = [theta_plv, alpha_plv, beta_plv, gamma_plv]
    # ground truth matrix using coh
    coh_combined_ground_truth_matrices = [theta_coh, alpha_coh, beta_coh, gamma_coh]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for ccorr)
    ccorr_theta_n_connections = np.zeros([16, 16])
    ccorr_alpha_n_connections = np.zeros([16, 16])
    ccorr_beta_n_connections = np.zeros([16, 16])
    ccorr_gamma_n_connections = np.zeros([16, 16])
    ccorr_combined_freq_n_connections = [
        ccorr_theta_n_connections,
        ccorr_alpha_n_connections,
        ccorr_beta_n_connections,
        ccorr_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for plv)
    plv_theta_n_connections = np.zeros([16, 16])
    plv_alpha_n_connections = np.zeros([16, 16])
    plv_beta_n_connections = np.zeros([16, 16])
    plv_gamma_n_connections = np.zeros([16, 16])
    plv_combined_freq_n_connections = [
        plv_theta_n_connections,
        plv_alpha_n_connections,
        plv_beta_n_connections,
        plv_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for coh)
    coh_theta_n_connections = np.zeros([16, 16])
    coh_alpha_n_connections = np.zeros([16, 16])
    coh_beta_n_connections = np.zeros([16, 16])
    coh_gamma_n_connections = np.zeros([16, 16])
    coh_combined_freq_n_connections = [
        coh_theta_n_connections,
        coh_alpha_n_connections,
        coh_beta_n_connections,
        coh_gamma_n_connections,
    ]

    ############ Implemanting Permutation test ################################

    # TODO Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

    ccorr_combined_freqs_electrode_pair_n_actual_score = []
    plv_combined_freqs_electrode_pair_n_actual_score = []
    coh_combined_freqs_electrode_pair_n_actual_score = []

    electrode_pair_n_actual_score_theta_ccorr = []
    electrode_pair_n_actual_score_alpha_ccorr = []
    electrode_pair_n_actual_score_beta_ccorr = []
    electrode_pair_n_actual_score_gamma_ccorr = []

    electrode_pair_n_actual_score_theta_plv = []
    electrode_pair_n_actual_score_alpha_plv = []
    electrode_pair_n_actual_score_beta_plv = []
    electrode_pair_n_actual_score_gamma_plv = []

    electrode_pair_n_actual_score_theta_coh = []
    electrode_pair_n_actual_score_alpha_coh = []
    electrode_pair_n_actual_score_beta_coh = []
    electrode_pair_n_actual_score_gamma_coh = []

    for participant1_channel in range(len(ch_names)):
        for participant2_channel in range(len(ch_names)):

            # epoch1 should just be a specific electrode e.g. FP1.
            # epoch1 = load_epoch_S1.pick_channels('FP1') or something. <- note this is not correct, it's just an idea
            epoch1 = load_epoch_S1.get_data(picks=ch_names[participant1_channel])
            epoch2 = load_epoch_S2.get_data(picks=ch_names[participant2_channel])

            rng = np.random.default_rng(42)  # set a random seed

            # Permute for several times as defined above
            # TODO Change this to 80 for real one
            n_perms = 150

            k_ccorr_theta_permuted = (
                []
            )  # initialising list that will store all the ccor values (ccorr value range of 0 to +1). its length should equal n_perms
            k_ccorr_alpha_permuted = []
            k_ccorr_beta_permuted = []
            k_ccorr_gamma_permuted = []

            k_plv_theta_permuted = (
                []
            )  # initialising list that will store all the plv values (plv value range of 0 to +1). its length should equal n_perms
            k_plv_alpha_permuted = []
            k_plv_beta_permuted = []
            k_plv_gamma_permuted = []

            k_coh_theta_permuted = (
                []
            )  # initialising list that will store all the coh values (coh value range of 0 to +1). its length should equal n_perms
            k_coh_alpha_permuted = []
            k_coh_beta_permuted = []
            k_coh_gamma_permuted = []

            # for each iterations, undergo permutation and calculate ccorr or plv or coh
            for iperm in range(n_perms):

                # for participant 1
                perm1 = rng.permutation(
                    len(epoch1)
                )  # randomising indices without replacement
                epoch1_in_permuted_order = [
                    epoch1[i] for i in perm1
                ]  # index the epoch to get permuted epochs

                # for participant 2
                perm2 = rng.permutation(len(epoch2))
                epoch2_in_permuted_order = [epoch1[i] for i in perm2]

                # combine the two permuted epochs together
                data_inter_permuted = np.array([epoch1, epoch2])

                # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                complex_signal = analyses.compute_freq_bands(
                    data_inter_permuted, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
                result_plv = analyses.compute_sync(complex_signal, mode="plv")
                result_coh = analyses.compute_sync(complex_signal, mode="coh")

                # comparing 1 channel of participant 1 against 1 channel of participant 2
                # n_ch is 1 instead of 16 because we are finding connection of 1 electrode and 1 other electrode each time.
                n_ch = 1
                # slice the result array to seperate into different frequencies. do this for 'ccorr', 'plv' and 'coh'
                (
                    ccorr_theta_permuted,
                    ccorr_alpha_permuted,
                    ccorr_beta_permuted,
                    ccorr_gamma_permuted,
                ) = result_ccorr[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    plv_theta_permuted,
                    plv_alpha_permuted,
                    plv_beta_permuted,
                    plv_gamma_permuted,
                ) = result_plv[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    coh_theta_permuted,
                    coh_alpha_permuted,
                    coh_beta_permuted,
                    coh_gamma_permuted,
                ) = result_coh[:, 0:n_ch, n_ch : 2 * n_ch]

                # append the ccorr value to the corresponding list
                k_ccorr_theta_permuted.append(ccorr_theta_permuted)
                k_ccorr_alpha_permuted.append(ccorr_alpha_permuted)
                k_ccorr_beta_permuted.append(ccorr_beta_permuted)
                k_ccorr_gamma_permuted.append(ccorr_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta ccorr permutation scores, 2nd index stores alpha ccorr permutation scores, etc.
                combined_k_ccorr_frequency_permuted = [
                    k_ccorr_theta_permuted,
                    k_ccorr_alpha_permuted,
                    k_ccorr_beta_permuted,
                    k_ccorr_gamma_permuted,
                ]

                # append the plv value to the corresponding list
                k_plv_theta_permuted.append(plv_theta_permuted)
                k_plv_alpha_permuted.append(plv_alpha_permuted)
                k_plv_beta_permuted.append(plv_beta_permuted)
                k_plv_gamma_permuted.append(plv_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta plv permutation scores, 2nd index stores alpha plv permutation scores, etc.
                combined_k_plv_frequency_permuted = [
                    k_plv_theta_permuted,
                    k_plv_alpha_permuted,
                    k_plv_beta_permuted,
                    k_plv_gamma_permuted,
                ]

                # append the coh value to the corresponding list
                k_coh_theta_permuted.append(coh_theta_permuted)
                k_coh_alpha_permuted.append(coh_alpha_permuted)
                k_coh_beta_permuted.append(coh_beta_permuted)
                k_coh_gamma_permuted.append(coh_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta coh permutation scores, 2nd index stores alpha coh permutation scores, etc.
                combined_k_coh_frequency_permuted = [
                    k_coh_theta_permuted,
                    k_coh_alpha_permuted,
                    k_coh_beta_permuted,
                    k_coh_gamma_permuted,
                ]

            # iterate each theta, alpha, beta, gamma
            for iterate_each_freq in range(
                len(combined_k_ccorr_frequency_permuted)
            ):  # 4 because we have classified frequency range into 4 types: theta, alpha, beta, gamma

                # Calculate p value
                z_value = 1.96  # equivalent to p value of 0.05

                # calculate mean and standard deviation for each frequency band using ccorr
                ccorr_mean_permuted = np.mean(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_std_permuted = np.std(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_z_score = (
                    ccorr_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - ccorr_mean_permuted
                ) / ccorr_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(ccorr_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    ccorr_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using plv
                plv_mean_permuted = np.mean(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_std_permuted = np.std(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_z_score = (
                    plv_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - plv_mean_permuted
                ) / plv_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(plv_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    plv_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using coh
                coh_mean_permuted = np.mean(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_std_permuted = np.std(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_z_score = (
                    coh_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - coh_mean_permuted
                ) / coh_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(coh_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    coh_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

    # convert the 4 x 16 x 16 array into a list (marker significant connection matrix)
    ccorr_combined_freq_n_connections_list = list(ccorr_combined_freq_n_connections)
    plv_combined_freq_n_connections_list = list(plv_combined_freq_n_connections)
    coh_combined_freq_n_connections_list = list(coh_combined_freq_n_connections)

    # Progress getting actual score from marker significant connection matrix (ccorr)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_ccorr = []
    alpha_sig_electrode_pair_ccorr = []
    beta_sig_electrode_pair_ccorr = []
    gamma_sig_electrode_pair_ccorr = []

    for idx_freq in range(len(ccorr_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(ccorr_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                ccorr_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if ccorr_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = ccorr_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_ccorr = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for ccorr)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )

    # Create main list that contains all the above 4 lists of frequency.
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        beta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_ccorr
    )

    # Progress getting actual score from marker significant connection matrix (plv)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_plv = []
    alpha_sig_electrode_pair_plv = []
    beta_sig_electrode_pair_plv = []
    gamma_sig_electrode_pair_plv = []

    for idx_freq in range(len(plv_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(plv_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                plv_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if plv_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = plv_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_plv = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for plv)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )

    # Create main list that contains all the above 4 lists of frequency.
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_plv)
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_plv
    )

    # Progress getting actual score from marker significant connection matrix (coh)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_coh = []
    alpha_sig_electrode_pair_coh = []
    beta_sig_electrode_pair_coh = []
    gamma_sig_electrode_pair_coh = []

    for idx_freq in range(len(coh_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(coh_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                coh_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if coh_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = coh_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_coh = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for coh)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )

    # Create main list that contains all the above 4 lists of frequency.
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_coh)
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_coh
    )

    # So there will be 3 main containers (actual score of ccor, plv, coh). Each of them has 4 lists (theta, alpha, beta, and gamma)

    # save ccorr connection data for a pair
    saved_filename1 = (
        saved_directory
        + "Pre_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of ccorr for a pair
    saved_actual_score_filename1 = (
        saved_directory
        + "Pre_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save plv connection data for a pair
    saved_filename2 = (
        saved_directory
        + "Pre_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of plv for a pair
    saved_actual_score_filename2 = (
        saved_directory
        + "Pre_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save coh connection data for a pair
    saved_filename3 = (
        saved_directory
        + "Pre_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of coh for a pair
    saved_actual_score_filename3 = (
        saved_directory
        + "Pre_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# %% [markdown]
# ### Statistical analysis (direct_post)

# %%
# TODO Where preprocessed files are stored (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
path_2_preproc_direct_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/direct_post/"

# TODO Directory to save significant connections (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
saved_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_post/"

# IMPORTANT ! how many permutations you want
n_perms = 150

# List files that are contained in a specified directory
list_of_files = os.listdir(path_2_preproc_direct_post)

# Change working directory
os.chdir(path_2_preproc_direct_post)

# To loop subject number.
begin = 0
end = len(list_of_files)
step = 2

freq_bands = {
    "Theta": [4, 7],
    "Alpha": [7.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 40],
}
freq_bands = OrderedDict(freq_bands)

ch_names = [
    "FP1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
ch_types = ["eeg"] * 16
info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)


for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    filename1 = list_of_files[i]
    filename2 = list_of_files[i + 1]

    # Load preprocessed epochs
    load_epoch_S1 = mne.read_epochs(filename1, preload=True)
    load_epoch_S2 = mne.read_epochs(filename2, preload=True)

    # Equalize number of epochs
    mne.epochs.equalize_epoch_counts([load_epoch_S1, load_epoch_S2])

    sampling_rate = load_epoch_S1.info["sfreq"]

    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        load_epoch_S1, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    psd2 = analyses.pow(
        load_epoch_S2, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    # ### Connectivity
    # with ICA
    data_inter = np.array([load_epoch_S1, load_epoch_S2])
    result_intra = []

    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

    # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
    ground_result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
    ground_result_plv = analyses.compute_sync(complex_signal, mode="plv")
    ground_result_coh = analyses.compute_sync(complex_signal, mode="coh")

    # Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    n_ch = len(load_epoch_S1.info["ch_names"])
    theta_ccorr, alpha_ccorr, beta_ccorr, gamma_ccorr = ground_result_ccorr[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_plv, alpha_plv, beta_plv, gamma_plv = ground_result_plv[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_coh, alpha_coh, beta_coh, gamma_coh = ground_result_coh[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]

    # TODO Get the type of data this and change the type to ndarray in later down the section
    # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
    # ground truth matrix using ccorr
    ccorr_combined_ground_truth_matrices = [
        theta_ccorr,
        alpha_ccorr,
        beta_ccorr,
        gamma_ccorr,
    ]
    # ground truth matrix using plv
    plv_combined_ground_truth_matrices = [theta_plv, alpha_plv, beta_plv, gamma_plv]
    # ground truth matrix using coh
    coh_combined_ground_truth_matrices = [theta_coh, alpha_coh, beta_coh, gamma_coh]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for ccorr)
    ccorr_theta_n_connections = np.zeros([16, 16])
    ccorr_alpha_n_connections = np.zeros([16, 16])
    ccorr_beta_n_connections = np.zeros([16, 16])
    ccorr_gamma_n_connections = np.zeros([16, 16])
    ccorr_combined_freq_n_connections = [
        ccorr_theta_n_connections,
        ccorr_alpha_n_connections,
        ccorr_beta_n_connections,
        ccorr_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for plv)
    plv_theta_n_connections = np.zeros([16, 16])
    plv_alpha_n_connections = np.zeros([16, 16])
    plv_beta_n_connections = np.zeros([16, 16])
    plv_gamma_n_connections = np.zeros([16, 16])
    plv_combined_freq_n_connections = [
        plv_theta_n_connections,
        plv_alpha_n_connections,
        plv_beta_n_connections,
        plv_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for coh)
    coh_theta_n_connections = np.zeros([16, 16])
    coh_alpha_n_connections = np.zeros([16, 16])
    coh_beta_n_connections = np.zeros([16, 16])
    coh_gamma_n_connections = np.zeros([16, 16])
    coh_combined_freq_n_connections = [
        coh_theta_n_connections,
        coh_alpha_n_connections,
        coh_beta_n_connections,
        coh_gamma_n_connections,
    ]

    ############ Implemanting Permutation test ################################

    # TODO Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

    ccorr_combined_freqs_electrode_pair_n_actual_score = []
    plv_combined_freqs_electrode_pair_n_actual_score = []
    coh_combined_freqs_electrode_pair_n_actual_score = []

    electrode_pair_n_actual_score_theta_ccorr = []
    electrode_pair_n_actual_score_alpha_ccorr = []
    electrode_pair_n_actual_score_beta_ccorr = []
    electrode_pair_n_actual_score_gamma_ccorr = []

    electrode_pair_n_actual_score_theta_plv = []
    electrode_pair_n_actual_score_alpha_plv = []
    electrode_pair_n_actual_score_beta_plv = []
    electrode_pair_n_actual_score_gamma_plv = []

    electrode_pair_n_actual_score_theta_coh = []
    electrode_pair_n_actual_score_alpha_coh = []
    electrode_pair_n_actual_score_beta_coh = []
    electrode_pair_n_actual_score_gamma_coh = []

    for participant1_channel in range(len(ch_names)):
        for participant2_channel in range(len(ch_names)):

            # epoch1 should just be a specific electrode e.g. FP1.
            # epoch1 = load_epoch_S1.pick_channels('FP1') or something. <- note this is not correct, it's just an idea
            epoch1 = load_epoch_S1.get_data(picks=ch_names[participant1_channel])
            epoch2 = load_epoch_S2.get_data(picks=ch_names[participant2_channel])

            rng = np.random.default_rng(42)  # set a random seed

            # Permute for several times as defined above
            # TODO Change this to 80 for real one
            n_perms = 150

            k_ccorr_theta_permuted = (
                []
            )  # initialising list that will store all the ccor values (ccorr value range of 0 to +1). its length should equal n_perms
            k_ccorr_alpha_permuted = []
            k_ccorr_beta_permuted = []
            k_ccorr_gamma_permuted = []

            k_plv_theta_permuted = (
                []
            )  # initialising list that will store all the plv values (plv value range of 0 to +1). its length should equal n_perms
            k_plv_alpha_permuted = []
            k_plv_beta_permuted = []
            k_plv_gamma_permuted = []

            k_coh_theta_permuted = (
                []
            )  # initialising list that will store all the coh values (coh value range of 0 to +1). its length should equal n_perms
            k_coh_alpha_permuted = []
            k_coh_beta_permuted = []
            k_coh_gamma_permuted = []

            # for each iterations, undergo permutation and calculate ccorr or plv or coh
            for iperm in range(n_perms):

                # for participant 1
                perm1 = rng.permutation(
                    len(epoch1)
                )  # randomising indices without replacement
                epoch1_in_permuted_order = [
                    epoch1[i] for i in perm1
                ]  # index the epoch to get permuted epochs

                # for participant 2
                perm2 = rng.permutation(len(epoch2))
                epoch2_in_permuted_order = [epoch1[i] for i in perm2]

                # combine the two permuted epochs together
                data_inter_permuted = np.array([epoch1, epoch2])

                # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                complex_signal = analyses.compute_freq_bands(
                    data_inter_permuted, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
                result_plv = analyses.compute_sync(complex_signal, mode="plv")
                result_coh = analyses.compute_sync(complex_signal, mode="coh")

                # comparing 1 channel of participant 1 against 1 channel of participant 2
                # n_ch is 1 instead of 16 because we are finding connection of 1 electrode and 1 other electrode each time.
                n_ch = 1
                # slice the result array to seperate into different frequencies. do this for 'ccorr', 'plv' and 'coh'
                (
                    ccorr_theta_permuted,
                    ccorr_alpha_permuted,
                    ccorr_beta_permuted,
                    ccorr_gamma_permuted,
                ) = result_ccorr[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    plv_theta_permuted,
                    plv_alpha_permuted,
                    plv_beta_permuted,
                    plv_gamma_permuted,
                ) = result_plv[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    coh_theta_permuted,
                    coh_alpha_permuted,
                    coh_beta_permuted,
                    coh_gamma_permuted,
                ) = result_coh[:, 0:n_ch, n_ch : 2 * n_ch]

                # append the ccorr value to the corresponding list
                k_ccorr_theta_permuted.append(ccorr_theta_permuted)
                k_ccorr_alpha_permuted.append(ccorr_alpha_permuted)
                k_ccorr_beta_permuted.append(ccorr_beta_permuted)
                k_ccorr_gamma_permuted.append(ccorr_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta ccorr permutation scores, 2nd index stores alpha ccorr permutation scores, etc.
                combined_k_ccorr_frequency_permuted = [
                    k_ccorr_theta_permuted,
                    k_ccorr_alpha_permuted,
                    k_ccorr_beta_permuted,
                    k_ccorr_gamma_permuted,
                ]

                # append the plv value to the corresponding list
                k_plv_theta_permuted.append(plv_theta_permuted)
                k_plv_alpha_permuted.append(plv_alpha_permuted)
                k_plv_beta_permuted.append(plv_beta_permuted)
                k_plv_gamma_permuted.append(plv_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta plv permutation scores, 2nd index stores alpha plv permutation scores, etc.
                combined_k_plv_frequency_permuted = [
                    k_plv_theta_permuted,
                    k_plv_alpha_permuted,
                    k_plv_beta_permuted,
                    k_plv_gamma_permuted,
                ]

                # append the coh value to the corresponding list
                k_coh_theta_permuted.append(coh_theta_permuted)
                k_coh_alpha_permuted.append(coh_alpha_permuted)
                k_coh_beta_permuted.append(coh_beta_permuted)
                k_coh_gamma_permuted.append(coh_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta coh permutation scores, 2nd index stores alpha coh permutation scores, etc.
                combined_k_coh_frequency_permuted = [
                    k_coh_theta_permuted,
                    k_coh_alpha_permuted,
                    k_coh_beta_permuted,
                    k_coh_gamma_permuted,
                ]

            # iterate each theta, alpha, beta, gamma
            for iterate_each_freq in range(
                len(combined_k_ccorr_frequency_permuted)
            ):  # 4 because we have classified frequency range into 4 types: theta, alpha, beta, gamma

                # Calculate p value
                z_value = 1.96  # equivalent to p value of 0.05

                # calculate mean and standard deviation for each frequency band using ccorr
                ccorr_mean_permuted = np.mean(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_std_permuted = np.std(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_z_score = (
                    ccorr_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - ccorr_mean_permuted
                ) / ccorr_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(ccorr_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    ccorr_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using plv
                plv_mean_permuted = np.mean(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_std_permuted = np.std(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_z_score = (
                    plv_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - plv_mean_permuted
                ) / plv_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(plv_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    plv_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using coh
                coh_mean_permuted = np.mean(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_std_permuted = np.std(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_z_score = (
                    coh_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - coh_mean_permuted
                ) / coh_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(coh_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    coh_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

    # convert the 4 x 16 x 16 array into a list (marker significant connection matrix)
    ccorr_combined_freq_n_connections_list = list(ccorr_combined_freq_n_connections)
    plv_combined_freq_n_connections_list = list(plv_combined_freq_n_connections)
    coh_combined_freq_n_connections_list = list(coh_combined_freq_n_connections)

    # Progress getting actual score from marker significant connection matrix (ccorr)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_ccorr = []
    alpha_sig_electrode_pair_ccorr = []
    beta_sig_electrode_pair_ccorr = []
    gamma_sig_electrode_pair_ccorr = []

    for idx_freq in range(len(ccorr_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(ccorr_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                ccorr_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if ccorr_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = ccorr_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_ccorr = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for ccorr)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )

    # Create main list that contains all the above 4 lists of frequency.
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        beta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_ccorr
    )

    # Progress getting actual score from marker significant connection matrix (plv)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_plv = []
    alpha_sig_electrode_pair_plv = []
    beta_sig_electrode_pair_plv = []
    gamma_sig_electrode_pair_plv = []

    for idx_freq in range(len(plv_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(plv_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                plv_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if plv_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = plv_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_plv = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for plv)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )

    # Create main list that contains all the above 4 lists of frequency.
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_plv)
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_plv
    )

    # Progress getting actual score from marker significant connection matrix (coh)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_coh = []
    alpha_sig_electrode_pair_coh = []
    beta_sig_electrode_pair_coh = []
    gamma_sig_electrode_pair_coh = []

    for idx_freq in range(len(coh_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(coh_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                coh_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if coh_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = coh_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_coh = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for coh)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )

    # Create main list that contains all the above 4 lists of frequency.
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_coh)
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_coh
    )

    # So there will be 3 main containers (actual score of ccor, plv, coh). Each of them has 4 lists (theta, alpha, beta, and gamma)

    # save ccorr connection data for a pair
    saved_filename1 = (
        saved_directory
        + "Post_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of ccorr for a pair
    saved_actual_score_filename1 = (
        saved_directory
        + "Post_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save plv connection data for a pair
    saved_filename2 = (
        saved_directory
        + "Post_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of plv for a pair
    saved_actual_score_filename2 = (
        saved_directory
        + "Post_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save coh connection data for a pair
    saved_filename3 = (
        saved_directory
        + "Post_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of coh for a pair
    saved_actual_score_filename3 = (
        saved_directory
        + "Post_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

# %% [markdown]
# ### Statistical analysis (natural_pre)

# %%

# TODO Where preprocessed files are stored (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
path_2_preproc_natural_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/natural_pre/"

# TODO Directory to save significant connections (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
saved_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_pre/"

# IMPORTANT ! how many permutations you want
n_perms = 150

# List files that are contained in a specified directory
list_of_files = os.listdir(path_2_preproc_natural_pre)

# Change working directory
os.chdir(path_2_preproc_natural_pre)

# To loop subject number.
begin = 0
end = len(list_of_files)
step = 2

freq_bands = {
    "Theta": [4, 7],
    "Alpha": [7.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 40],
}
freq_bands = OrderedDict(freq_bands)

ch_names = [
    "FP1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
ch_types = ["eeg"] * 16
info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)


for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    filename1 = list_of_files[i]
    filename2 = list_of_files[i + 1]

    # Load preprocessed epochs
    load_epoch_S1 = mne.read_epochs(filename1, preload=True)
    load_epoch_S2 = mne.read_epochs(filename2, preload=True)

    # Equalize number of epochs
    mne.epochs.equalize_epoch_counts([load_epoch_S1, load_epoch_S2])

    sampling_rate = load_epoch_S1.info["sfreq"]

    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        load_epoch_S1, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    psd2 = analyses.pow(
        load_epoch_S2, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    # ### Connectivity
    # with ICA
    data_inter = np.array([load_epoch_S1, load_epoch_S2])
    result_intra = []

    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

    # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
    ground_result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
    ground_result_plv = analyses.compute_sync(complex_signal, mode="plv")
    ground_result_coh = analyses.compute_sync(complex_signal, mode="coh")

    # Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    n_ch = len(load_epoch_S1.info["ch_names"])
    theta_ccorr, alpha_ccorr, beta_ccorr, gamma_ccorr = ground_result_ccorr[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_plv, alpha_plv, beta_plv, gamma_plv = ground_result_plv[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_coh, alpha_coh, beta_coh, gamma_coh = ground_result_coh[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]

    # TODO Get the type of data this and change the type to ndarray in later down the section
    # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
    # ground truth matrix using ccorr
    ccorr_combined_ground_truth_matrices = [
        theta_ccorr,
        alpha_ccorr,
        beta_ccorr,
        gamma_ccorr,
    ]
    # ground truth matrix using plv
    plv_combined_ground_truth_matrices = [theta_plv, alpha_plv, beta_plv, gamma_plv]
    # ground truth matrix using coh
    coh_combined_ground_truth_matrices = [theta_coh, alpha_coh, beta_coh, gamma_coh]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for ccorr)
    ccorr_theta_n_connections = np.zeros([16, 16])
    ccorr_alpha_n_connections = np.zeros([16, 16])
    ccorr_beta_n_connections = np.zeros([16, 16])
    ccorr_gamma_n_connections = np.zeros([16, 16])
    ccorr_combined_freq_n_connections = [
        ccorr_theta_n_connections,
        ccorr_alpha_n_connections,
        ccorr_beta_n_connections,
        ccorr_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for plv)
    plv_theta_n_connections = np.zeros([16, 16])
    plv_alpha_n_connections = np.zeros([16, 16])
    plv_beta_n_connections = np.zeros([16, 16])
    plv_gamma_n_connections = np.zeros([16, 16])
    plv_combined_freq_n_connections = [
        plv_theta_n_connections,
        plv_alpha_n_connections,
        plv_beta_n_connections,
        plv_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for coh)
    coh_theta_n_connections = np.zeros([16, 16])
    coh_alpha_n_connections = np.zeros([16, 16])
    coh_beta_n_connections = np.zeros([16, 16])
    coh_gamma_n_connections = np.zeros([16, 16])
    coh_combined_freq_n_connections = [
        coh_theta_n_connections,
        coh_alpha_n_connections,
        coh_beta_n_connections,
        coh_gamma_n_connections,
    ]

    ############ Implemanting Permutation test ################################

    # TODO Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

    ccorr_combined_freqs_electrode_pair_n_actual_score = []
    plv_combined_freqs_electrode_pair_n_actual_score = []
    coh_combined_freqs_electrode_pair_n_actual_score = []

    electrode_pair_n_actual_score_theta_ccorr = []
    electrode_pair_n_actual_score_alpha_ccorr = []
    electrode_pair_n_actual_score_beta_ccorr = []
    electrode_pair_n_actual_score_gamma_ccorr = []

    electrode_pair_n_actual_score_theta_plv = []
    electrode_pair_n_actual_score_alpha_plv = []
    electrode_pair_n_actual_score_beta_plv = []
    electrode_pair_n_actual_score_gamma_plv = []

    electrode_pair_n_actual_score_theta_coh = []
    electrode_pair_n_actual_score_alpha_coh = []
    electrode_pair_n_actual_score_beta_coh = []
    electrode_pair_n_actual_score_gamma_coh = []

    for participant1_channel in range(len(ch_names)):
        for participant2_channel in range(len(ch_names)):

            # epoch1 should just be a specific electrode e.g. FP1.
            # epoch1 = load_epoch_S1.pick_channels('FP1') or something. <- note this is not correct, it's just an idea
            epoch1 = load_epoch_S1.get_data(picks=ch_names[participant1_channel])
            epoch2 = load_epoch_S2.get_data(picks=ch_names[participant2_channel])

            rng = np.random.default_rng(42)  # set a random seed

            # Permute for several times as defined above
            # TODO Change this to 80 for real one
            n_perms = 150

            k_ccorr_theta_permuted = (
                []
            )  # initialising list that will store all the ccor values (ccorr value range of 0 to +1). its length should equal n_perms
            k_ccorr_alpha_permuted = []
            k_ccorr_beta_permuted = []
            k_ccorr_gamma_permuted = []

            k_plv_theta_permuted = (
                []
            )  # initialising list that will store all the plv values (plv value range of 0 to +1). its length should equal n_perms
            k_plv_alpha_permuted = []
            k_plv_beta_permuted = []
            k_plv_gamma_permuted = []

            k_coh_theta_permuted = (
                []
            )  # initialising list that will store all the coh values (coh value range of 0 to +1). its length should equal n_perms
            k_coh_alpha_permuted = []
            k_coh_beta_permuted = []
            k_coh_gamma_permuted = []

            # for each iterations, undergo permutation and calculate ccorr or plv or coh
            for iperm in range(n_perms):

                # for participant 1
                perm1 = rng.permutation(
                    len(epoch1)
                )  # randomising indices without replacement
                epoch1_in_permuted_order = [
                    epoch1[i] for i in perm1
                ]  # index the epoch to get permuted epochs

                # for participant 2
                perm2 = rng.permutation(len(epoch2))
                epoch2_in_permuted_order = [epoch1[i] for i in perm2]

                # combine the two permuted epochs together
                data_inter_permuted = np.array([epoch1, epoch2])

                # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                complex_signal = analyses.compute_freq_bands(
                    data_inter_permuted, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
                result_plv = analyses.compute_sync(complex_signal, mode="plv")
                result_coh = analyses.compute_sync(complex_signal, mode="coh")

                # comparing 1 channel of participant 1 against 1 channel of participant 2
                # n_ch is 1 instead of 16 because we are finding connection of 1 electrode and 1 other electrode each time.
                n_ch = 1
                # slice the result array to seperate into different frequencies. do this for 'ccorr', 'plv' and 'coh'
                (
                    ccorr_theta_permuted,
                    ccorr_alpha_permuted,
                    ccorr_beta_permuted,
                    ccorr_gamma_permuted,
                ) = result_ccorr[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    plv_theta_permuted,
                    plv_alpha_permuted,
                    plv_beta_permuted,
                    plv_gamma_permuted,
                ) = result_plv[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    coh_theta_permuted,
                    coh_alpha_permuted,
                    coh_beta_permuted,
                    coh_gamma_permuted,
                ) = result_coh[:, 0:n_ch, n_ch : 2 * n_ch]

                # append the ccorr value to the corresponding list
                k_ccorr_theta_permuted.append(ccorr_theta_permuted)
                k_ccorr_alpha_permuted.append(ccorr_alpha_permuted)
                k_ccorr_beta_permuted.append(ccorr_beta_permuted)
                k_ccorr_gamma_permuted.append(ccorr_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta ccorr permutation scores, 2nd index stores alpha ccorr permutation scores, etc.
                combined_k_ccorr_frequency_permuted = [
                    k_ccorr_theta_permuted,
                    k_ccorr_alpha_permuted,
                    k_ccorr_beta_permuted,
                    k_ccorr_gamma_permuted,
                ]

                # append the plv value to the corresponding list
                k_plv_theta_permuted.append(plv_theta_permuted)
                k_plv_alpha_permuted.append(plv_alpha_permuted)
                k_plv_beta_permuted.append(plv_beta_permuted)
                k_plv_gamma_permuted.append(plv_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta plv permutation scores, 2nd index stores alpha plv permutation scores, etc.
                combined_k_plv_frequency_permuted = [
                    k_plv_theta_permuted,
                    k_plv_alpha_permuted,
                    k_plv_beta_permuted,
                    k_plv_gamma_permuted,
                ]

                # append the coh value to the corresponding list
                k_coh_theta_permuted.append(coh_theta_permuted)
                k_coh_alpha_permuted.append(coh_alpha_permuted)
                k_coh_beta_permuted.append(coh_beta_permuted)
                k_coh_gamma_permuted.append(coh_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta coh permutation scores, 2nd index stores alpha coh permutation scores, etc.
                combined_k_coh_frequency_permuted = [
                    k_coh_theta_permuted,
                    k_coh_alpha_permuted,
                    k_coh_beta_permuted,
                    k_coh_gamma_permuted,
                ]

            # iterate each theta, alpha, beta, gamma
            for iterate_each_freq in range(
                len(combined_k_ccorr_frequency_permuted)
            ):  # 4 because we have classified frequency range into 4 types: theta, alpha, beta, gamma

                # Calculate p value
                z_value = 1.96  # equivalent to p value of 0.05

                # calculate mean and standard deviation for each frequency band using ccorr
                ccorr_mean_permuted = np.mean(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_std_permuted = np.std(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_z_score = (
                    ccorr_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - ccorr_mean_permuted
                ) / ccorr_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(ccorr_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    ccorr_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using plv
                plv_mean_permuted = np.mean(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_std_permuted = np.std(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_z_score = (
                    plv_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - plv_mean_permuted
                ) / plv_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(plv_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    plv_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using coh
                coh_mean_permuted = np.mean(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_std_permuted = np.std(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_z_score = (
                    coh_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - coh_mean_permuted
                ) / coh_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(coh_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    coh_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

    # convert the 4 x 16 x 16 array into a list (marker significant connection matrix)
    ccorr_combined_freq_n_connections_list = list(ccorr_combined_freq_n_connections)
    plv_combined_freq_n_connections_list = list(plv_combined_freq_n_connections)
    coh_combined_freq_n_connections_list = list(coh_combined_freq_n_connections)

    # Progress getting actual score from marker significant connection matrix (ccorr)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_ccorr = []
    alpha_sig_electrode_pair_ccorr = []
    beta_sig_electrode_pair_ccorr = []
    gamma_sig_electrode_pair_ccorr = []

    for idx_freq in range(len(ccorr_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(ccorr_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                ccorr_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if ccorr_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = ccorr_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_ccorr = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for ccorr)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )

    # Create main list that contains all the above 4 lists of frequency.
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        beta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_ccorr
    )

    # Progress getting actual score from marker significant connection matrix (plv)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_plv = []
    alpha_sig_electrode_pair_plv = []
    beta_sig_electrode_pair_plv = []
    gamma_sig_electrode_pair_plv = []

    for idx_freq in range(len(plv_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(plv_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                plv_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if plv_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = plv_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_plv = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for plv)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )

    # Create main list that contains all the above 4 lists of frequency.
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_plv)
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_plv
    )

    # Progress getting actual score from marker significant connection matrix (coh)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_coh = []
    alpha_sig_electrode_pair_coh = []
    beta_sig_electrode_pair_coh = []
    gamma_sig_electrode_pair_coh = []

    for idx_freq in range(len(coh_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(coh_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                coh_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if coh_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = coh_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_coh = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for coh)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )

    # Create main list that contains all the above 4 lists of frequency.
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_coh)
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_coh
    )

    # So there will be 3 main containers (actual score of ccor, plv, coh). Each of them has 4 lists (theta, alpha, beta, and gamma)

    # save ccorr connection data for a pair
    saved_filename1 = (
        saved_directory
        + "Pre_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of ccorr for a pair
    saved_actual_score_filename1 = (
        saved_directory
        + "Pre_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save plv connection data for a pair
    saved_filename2 = (
        saved_directory
        + "Pre_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of plv for a pair
    saved_actual_score_filename2 = (
        saved_directory
        + "Pre_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save coh connection data for a pair
    saved_filename3 = (
        saved_directory
        + "Pre_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of coh for a pair
    saved_actual_score_filename3 = (
        saved_directory
        + "Pre_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

# %% [markdown]
# ### Statistical analysis (natural_post)

# %%
# TODO Where preprocessed files are stored (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
path_2_preproc_natural_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/natural_post/"

# TODO Directory to save significant connections (REMEMBER ! Different eye condition (pre/pro) requires a unique directory)
saved_directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_post/"

# IMPORTANT ! how many permutations you want
n_perms = 150

# List files that are contained in a specified directory
list_of_files = os.listdir(path_2_preproc_natural_post)

# Change working directory
os.chdir(path_2_preproc_natural_post)

# To loop subject number.
begin = 0
end = len(list_of_files)
step = 2

freq_bands = {
    "Theta": [4, 7],
    "Alpha": [7.5, 13],
    "Beta": [13.5, 29.5],
    "Gamma": [30, 40],
}
freq_bands = OrderedDict(freq_bands)

ch_names = [
    "FP1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]
ch_types = ["eeg"] * 16
info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)


for i in tqdm(
    range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
):

    filename1 = list_of_files[i]
    filename2 = list_of_files[i + 1]

    # Load preprocessed epochs
    load_epoch_S1 = mne.read_epochs(filename1, preload=True)
    load_epoch_S2 = mne.read_epochs(filename2, preload=True)

    # Equalize number of epochs
    mne.epochs.equalize_epoch_counts([load_epoch_S1, load_epoch_S2])

    sampling_rate = load_epoch_S1.info["sfreq"]

    # Analysing data Welch Power Spectral Density. Here for ex, the frequency-band-of-interest,
    # frequencies for which power spectral density is actually computed are returned in freq_list,and PSD values are averaged across epochs
    # Frequencies = Theta - Gamma (fmin, fmax) - kindly see the freq_bands

    psd1 = analyses.pow(
        load_epoch_S1, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    psd2 = analyses.pow(
        load_epoch_S2, fmin=4, fmax=40, n_fft=125, n_per_seg=60, epochs_average=True
    )
    data_psd = np.array([psd1.psd, psd2.psd])

    # ### Connectivity
    # with ICA
    data_inter = np.array([load_epoch_S1, load_epoch_S2])
    result_intra = []

    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

    # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
    ground_result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
    ground_result_plv = analyses.compute_sync(complex_signal, mode="plv")
    ground_result_coh = analyses.compute_sync(complex_signal, mode="coh")

    # Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    n_ch = len(load_epoch_S1.info["ch_names"])
    theta_ccorr, alpha_ccorr, beta_ccorr, gamma_ccorr = ground_result_ccorr[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_plv, alpha_plv, beta_plv, gamma_plv = ground_result_plv[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]
    theta_coh, alpha_coh, beta_coh, gamma_coh = ground_result_coh[
        :, 0:n_ch, n_ch : 2 * n_ch
    ]

    # TODO Get the type of data this and change the type to ndarray in later down the section
    # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
    # ground truth matrix using ccorr
    ccorr_combined_ground_truth_matrices = [
        theta_ccorr,
        alpha_ccorr,
        beta_ccorr,
        gamma_ccorr,
    ]
    # ground truth matrix using plv
    plv_combined_ground_truth_matrices = [theta_plv, alpha_plv, beta_plv, gamma_plv]
    # ground truth matrix using coh
    coh_combined_ground_truth_matrices = [theta_coh, alpha_coh, beta_coh, gamma_coh]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for ccorr)
    ccorr_theta_n_connections = np.zeros([16, 16])
    ccorr_alpha_n_connections = np.zeros([16, 16])
    ccorr_beta_n_connections = np.zeros([16, 16])
    ccorr_gamma_n_connections = np.zeros([16, 16])
    ccorr_combined_freq_n_connections = [
        ccorr_theta_n_connections,
        ccorr_alpha_n_connections,
        ccorr_beta_n_connections,
        ccorr_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for plv)
    plv_theta_n_connections = np.zeros([16, 16])
    plv_alpha_n_connections = np.zeros([16, 16])
    plv_beta_n_connections = np.zeros([16, 16])
    plv_gamma_n_connections = np.zeros([16, 16])
    plv_combined_freq_n_connections = [
        plv_theta_n_connections,
        plv_alpha_n_connections,
        plv_beta_n_connections,
        plv_gamma_n_connections,
    ]

    # array that stores 1 or 0 depending on whether the connection between two electrodes are significant (for coh)
    coh_theta_n_connections = np.zeros([16, 16])
    coh_alpha_n_connections = np.zeros([16, 16])
    coh_beta_n_connections = np.zeros([16, 16])
    coh_gamma_n_connections = np.zeros([16, 16])
    coh_combined_freq_n_connections = [
        coh_theta_n_connections,
        coh_alpha_n_connections,
        coh_beta_n_connections,
        coh_gamma_n_connections,
    ]

    ############ Implemanting Permutation test ################################

    # TODO Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

    ccorr_combined_freqs_electrode_pair_n_actual_score = []
    plv_combined_freqs_electrode_pair_n_actual_score = []
    coh_combined_freqs_electrode_pair_n_actual_score = []

    electrode_pair_n_actual_score_theta_ccorr = []
    electrode_pair_n_actual_score_alpha_ccorr = []
    electrode_pair_n_actual_score_beta_ccorr = []
    electrode_pair_n_actual_score_gamma_ccorr = []

    electrode_pair_n_actual_score_theta_plv = []
    electrode_pair_n_actual_score_alpha_plv = []
    electrode_pair_n_actual_score_beta_plv = []
    electrode_pair_n_actual_score_gamma_plv = []

    electrode_pair_n_actual_score_theta_coh = []
    electrode_pair_n_actual_score_alpha_coh = []
    electrode_pair_n_actual_score_beta_coh = []
    electrode_pair_n_actual_score_gamma_coh = []

    for participant1_channel in range(len(ch_names)):
        for participant2_channel in range(len(ch_names)):

            # epoch1 should just be a specific electrode e.g. FP1.
            # epoch1 = load_epoch_S1.pick_channels('FP1') or something. <- note this is not correct, it's just an idea
            epoch1 = load_epoch_S1.get_data(picks=ch_names[participant1_channel])
            epoch2 = load_epoch_S2.get_data(picks=ch_names[participant2_channel])

            rng = np.random.default_rng(42)  # set a random seed

            # Permute for several times as defined above
            # TODO Change this to 80 for real one
            n_perms = 150

            k_ccorr_theta_permuted = (
                []
            )  # initialising list that will store all the ccor values (ccorr value range of 0 to +1). its length should equal n_perms
            k_ccorr_alpha_permuted = []
            k_ccorr_beta_permuted = []
            k_ccorr_gamma_permuted = []

            k_plv_theta_permuted = (
                []
            )  # initialising list that will store all the plv values (plv value range of 0 to +1). its length should equal n_perms
            k_plv_alpha_permuted = []
            k_plv_beta_permuted = []
            k_plv_gamma_permuted = []

            k_coh_theta_permuted = (
                []
            )  # initialising list that will store all the coh values (coh value range of 0 to +1). its length should equal n_perms
            k_coh_alpha_permuted = []
            k_coh_beta_permuted = []
            k_coh_gamma_permuted = []

            # for each iterations, undergo permutation and calculate ccorr or plv or coh
            for iperm in range(n_perms):

                # for participant 1
                perm1 = rng.permutation(
                    len(epoch1)
                )  # randomising indices without replacement
                epoch1_in_permuted_order = [
                    epoch1[i] for i in perm1
                ]  # index the epoch to get permuted epochs

                # for participant 2
                perm2 = rng.permutation(len(epoch2))
                epoch2_in_permuted_order = [epoch1[i] for i in perm2]

                # combine the two permuted epochs together
                data_inter_permuted = np.array([epoch1, epoch2])

                # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                complex_signal = analyses.compute_freq_bands(
                    data_inter_permuted, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                result_ccorr = analyses.compute_sync(complex_signal, mode="ccorr")
                result_plv = analyses.compute_sync(complex_signal, mode="plv")
                result_coh = analyses.compute_sync(complex_signal, mode="coh")

                # comparing 1 channel of participant 1 against 1 channel of participant 2
                # n_ch is 1 instead of 16 because we are finding connection of 1 electrode and 1 other electrode each time.
                n_ch = 1
                # slice the result array to seperate into different frequencies. do this for 'ccorr', 'plv' and 'coh'
                (
                    ccorr_theta_permuted,
                    ccorr_alpha_permuted,
                    ccorr_beta_permuted,
                    ccorr_gamma_permuted,
                ) = result_ccorr[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    plv_theta_permuted,
                    plv_alpha_permuted,
                    plv_beta_permuted,
                    plv_gamma_permuted,
                ) = result_plv[:, 0:n_ch, n_ch : 2 * n_ch]
                (
                    coh_theta_permuted,
                    coh_alpha_permuted,
                    coh_beta_permuted,
                    coh_gamma_permuted,
                ) = result_coh[:, 0:n_ch, n_ch : 2 * n_ch]

                # append the ccorr value to the corresponding list
                k_ccorr_theta_permuted.append(ccorr_theta_permuted)
                k_ccorr_alpha_permuted.append(ccorr_alpha_permuted)
                k_ccorr_beta_permuted.append(ccorr_beta_permuted)
                k_ccorr_gamma_permuted.append(ccorr_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta ccorr permutation scores, 2nd index stores alpha ccorr permutation scores, etc.
                combined_k_ccorr_frequency_permuted = [
                    k_ccorr_theta_permuted,
                    k_ccorr_alpha_permuted,
                    k_ccorr_beta_permuted,
                    k_ccorr_gamma_permuted,
                ]

                # append the plv value to the corresponding list
                k_plv_theta_permuted.append(plv_theta_permuted)
                k_plv_alpha_permuted.append(plv_alpha_permuted)
                k_plv_beta_permuted.append(plv_beta_permuted)
                k_plv_gamma_permuted.append(plv_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta plv permutation scores, 2nd index stores alpha plv permutation scores, etc.
                combined_k_plv_frequency_permuted = [
                    k_plv_theta_permuted,
                    k_plv_alpha_permuted,
                    k_plv_beta_permuted,
                    k_plv_gamma_permuted,
                ]

                # append the coh value to the corresponding list
                k_coh_theta_permuted.append(coh_theta_permuted)
                k_coh_alpha_permuted.append(coh_alpha_permuted)
                k_coh_beta_permuted.append(coh_beta_permuted)
                k_coh_gamma_permuted.append(coh_gamma_permuted)
                # list of 4 lists stored. 1st index stores theta coh permutation scores, 2nd index stores alpha coh permutation scores, etc.
                combined_k_coh_frequency_permuted = [
                    k_coh_theta_permuted,
                    k_coh_alpha_permuted,
                    k_coh_beta_permuted,
                    k_coh_gamma_permuted,
                ]

            # iterate each theta, alpha, beta, gamma
            for iterate_each_freq in range(
                len(combined_k_ccorr_frequency_permuted)
            ):  # 4 because we have classified frequency range into 4 types: theta, alpha, beta, gamma

                # Calculate p value
                z_value = 1.96  # equivalent to p value of 0.05

                # calculate mean and standard deviation for each frequency band using ccorr
                ccorr_mean_permuted = np.mean(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_std_permuted = np.std(
                    combined_k_ccorr_frequency_permuted[iterate_each_freq]
                )
                ccorr_z_score = (
                    ccorr_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - ccorr_mean_permuted
                ) / ccorr_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(ccorr_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    ccorr_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using plv
                plv_mean_permuted = np.mean(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_std_permuted = np.std(
                    combined_k_plv_frequency_permuted[iterate_each_freq]
                )
                plv_z_score = (
                    plv_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - plv_mean_permuted
                ) / plv_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(plv_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    plv_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

                # calculate mean and standard deviation for each frequency band using coh
                coh_mean_permuted = np.mean(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_std_permuted = np.std(
                    combined_k_coh_frequency_permuted[iterate_each_freq]
                )
                coh_z_score = (
                    coh_combined_ground_truth_matrices[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel]
                    - coh_mean_permuted
                ) / coh_std_permuted
                # checking if the z score is greater than z value (p < .05) i.e. if the connection is significant or not. this serves as a marker of significant connection between two electrodes
                if np.abs(coh_z_score) > z_value:
                    # assign value 1 if the connection between ch_names[participant1_channel] and  ch_names[participant2_channel] is significant, otherwise 0 by default
                    # this is a 4 x 16 x 16 array. 1st index identifies which frequency band it is (e.g. theta, alpha, beta, gamma), and 2nd index represent the electrode of first participant, and 3rd index represent the electrode of second participant
                    coh_combined_freq_n_connections[iterate_each_freq][
                        participant1_channel
                    ][participant2_channel] = 1

    # convert the 4 x 16 x 16 array into a list (marker significant connection matrix)
    ccorr_combined_freq_n_connections_list = list(ccorr_combined_freq_n_connections)
    plv_combined_freq_n_connections_list = list(plv_combined_freq_n_connections)
    coh_combined_freq_n_connections_list = list(coh_combined_freq_n_connections)

    # Progress getting actual score from marker significant connection matrix (ccorr)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_ccorr = []
    alpha_sig_electrode_pair_ccorr = []
    beta_sig_electrode_pair_ccorr = []
    gamma_sig_electrode_pair_ccorr = []

    for idx_freq in range(len(ccorr_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(ccorr_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                ccorr_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if ccorr_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = ccorr_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_ccorr = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for ccorr)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_ccorr.append(
                            electrode_pair_n_actual_score_ccorr
                        )

    # Create main list that contains all the above 4 lists of frequency.
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        beta_sig_electrode_pair_ccorr
    )
    ccorr_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_ccorr
    )

    # Progress getting actual score from marker significant connection matrix (plv)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_plv = []
    alpha_sig_electrode_pair_plv = []
    beta_sig_electrode_pair_plv = []
    gamma_sig_electrode_pair_plv = []

    for idx_freq in range(len(plv_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(plv_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                plv_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if plv_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = plv_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_plv = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for plv)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_plv.append(
                            electrode_pair_n_actual_score_plv
                        )

    # Create main list that contains all the above 4 lists of frequency.
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_plv
    )
    plv_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_plv)
    plv_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_plv
    )

    # Progress getting actual score from marker significant connection matrix (coh)
    # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
    theta_sig_electrode_pair_coh = []
    alpha_sig_electrode_pair_coh = []
    beta_sig_electrode_pair_coh = []
    gamma_sig_electrode_pair_coh = []

    for idx_freq in range(len(coh_combined_freq_n_connections)):
        # Iterate over row of matrix (16 x 16)
        for idx_row, row in enumerate(coh_combined_freq_n_connections[idx_freq]):
            # Iterate over column of matrix (16 x 16)
            for idx_col, col in enumerate(
                coh_combined_freq_n_connections[idx_freq][idx_row]
            ):
                if coh_combined_freq_n_connections[idx_freq][idx_row][idx_col] == 1:
                    idx_sig_connection = tuple([idx_row, idx_col])
                    # Get actual score
                    actual_score = coh_combined_ground_truth_matrices[idx_freq][
                        idx_row
                    ][idx_col]
                    # Get pair label of electorode
                    sig_electrode_pair_label = get_electrode_labels_connections(
                        idx_sig_connection
                    )

                    electrode_pair_n_actual_score_coh = {
                        sig_electrode_pair_label: actual_score
                    }

                    # Put that string into a unique list (theta, alpha, beta, or gamma list for coh)
                    if idx_freq == 0:
                        theta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 1:
                        alpha_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 2:
                        beta_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )
                    elif idx_freq == 3:
                        gamma_sig_electrode_pair_coh.append(
                            electrode_pair_n_actual_score_coh
                        )

    # Create main list that contains all the above 4 lists of frequency.
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        theta_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        alpha_sig_electrode_pair_coh
    )
    coh_combined_freqs_electrode_pair_n_actual_score.append(beta_sig_electrode_pair_coh)
    coh_combined_freqs_electrode_pair_n_actual_score.append(
        gamma_sig_electrode_pair_coh
    )

    # So there will be 3 main containers (actual score of ccor, plv, coh). Each of them has 4 lists (theta, alpha, beta, and gamma)

    # save ccorr connection data for a pair
    saved_filename1 = (
        saved_directory
        + "Post_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of ccorr for a pair
    saved_actual_score_filename1 = (
        saved_directory
        + "Post_ccorr_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename1, "wb") as handle:
        pickle.dump(
            ccorr_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save plv connection data for a pair
    saved_filename2 = (
        saved_directory
        + "Post_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of plv for a pair
    saved_actual_score_filename2 = (
        saved_directory
        + "Post_plv_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename2, "wb") as handle:
        pickle.dump(
            plv_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # save coh connection data for a pair
    saved_filename3 = (
        saved_directory
        + "Post_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_connection_data.pkl"
    )
    with open(saved_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freq_n_connections_list,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Save actual significant scores of coh for a pair
    saved_actual_score_filename3 = (
        saved_directory
        + "Post_coh_combined_pair_S"
        + str(i + 1)
        + "_and_S"
        + str(i + 2)
        + "_actual_score_data.pkl"
    )
    with open(saved_actual_score_filename3, "wb") as handle:
        pickle.dump(
            coh_combined_freqs_electrode_pair_n_actual_score,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )