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
#     display_name: hyperscanning2_redesign_new
#     language: python
#     name: python3
# ---

# %%
# ### Relevant packages
import os
import pickle
from collections import OrderedDict, namedtuple
from typing import List

import heartrate
import matplotlib.pyplot as plt
import mne
import numpy as np
import snoop
from hypyp import analyses, prep, stats
from pandas import read_pickle
from scipy.stats import pearsonr
from tqdm import tqdm

from phd_codes.EEG import stats
from phd_codes.EEG.label_converter import get_electrode_labels_connections
from phd_codes.questionnaire.questionnaire import Questionnaire


# %%
class Connections:
    """
    Class that is related to a number of significant connections of EEG data
    """

    def permut_sig_connection(
        self, preproc_files: str, sig_connection_path: str, n_permutations: int
    ):

        """ 
            * To find out which connections that are statistically significant. This will run permutation \
            out of 256 connections of each pair (16 x 16 electrodes). See **note** and **warning**.


            :param preproc_files: Path to where pre-processed files are stored. See **note**.
            :type preproc_files: str
            :param sig_connection_path: Path to where significant connections files will be stored. See **note**.
            :type sig_connection_path: str
            :param n_permutations: number of permutations
            :type n_permutations: int

            :returns: :literal:`*.pkl` files, each pair will have 6 files. See **note**.
            :rtype: :literal:`*.pkl`

            .. note:: 
                * objective: 
                    The permutation for finding significant inter-brain synchrony scores \
                    are computed with three different algorithms as listed below. \
                    Each pair of electrode is computed by using the following algorithms.
                        #. ccorr - circular correlation coefficient
                        #. coh - coherence
                        #. plv - phase locking value
                                    
            
                * parameters: 
                    :literal:`preproc_files`

                    * Different eye condition (pre/pro) requires a unique directory.
                        
                        
                    :literal:`sig_connection_path`

                    * Different eye condition (pre/pro) requires a unique directory.
                    * There will be 3 main containers (actual score of ccor, plv, coh).
                    * Each of them has 4 lists (theta, alpha, beta, and gamma).

                * returns: 
                    ``*.pkl`` files. Each pair will have 6 files as shown below as example :
                        #. Pre_ccorr_combined_pair_S1_and_S2_actual_score_data.pkl
                        #. Pre_ccorr_combined_pair_S1_and_S2_connection_data.pkl
                        #. Pre_coh_combined_pair_S1_and_S2_actual_score_data.pkl
                        #. Pre_coh_combined_pair_S1_and_S2_actual_score_data.pkl
                        #. Pre_plv_combined_pair_S1_and_S2_actual_score_data.pkl
                        #. Pre_plv_combined_pair_S1_and_S2_connection_data.pkl
            
            .. seealso::
                * For more updated supported connectivity measures (algorithm) in `HyPyP module. <https://hypyp.readthedocs.io/en/latest/API/analyses/#hypyp.analyses.compute_sync>`_
                * :meth:`count_sig_connections` . That function uses parameter, which is the output of the current :meth:`permut_sig_connection` function. 

            .. warning:: 
                The higher permutation, the longer it takes time to process.\
                The current experiment used **30 times** permutation that applies to every electrode pair.

    

        """

        heartrate.trace(browser=True)

        path_2_preproc_averted_pre = preproc_files

        saved_directory = sig_connection_path

        # IMPORTANT ! how many permutations you want
        n_perms = n_permutations

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

        with snoop:
            for i in tqdm(
                range(begin, end, step),
                desc="Please, listen 2 music & have some coffee...",
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
                    load_epoch_S1,
                    fmin=4,
                    fmax=40,
                    n_fft=125,
                    n_per_seg=60,
                    epochs_average=True,
                )
                psd2 = analyses.pow(
                    load_epoch_S2,
                    fmin=4,
                    fmax=40,
                    n_fft=125,
                    n_per_seg=60,
                    epochs_average=True,
                )
                data_psd = np.array([psd1.psd, psd2.psd])

                # ### Connectivity
                # with ICA
                data_inter = np.array([load_epoch_S1, load_epoch_S2])
                result_intra = []

                # Computing analytic signal per frequency band
                # With ICA (Compute complex signal, that will be used as input for calculating connectivity, eg. power-correlation score)
                complex_signal = analyses.compute_freq_bands(
                    data_inter, sampling_rate, freq_bands
                )

                # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv' and 'coh'
                ground_result_ccorr = analyses.compute_sync(
                    complex_signal, mode="ccorr"
                )
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

                # Get the type of data this and change the type to ndarray in later down the section
                # So that it can be the same type of data ndarray of ground truth and ndarray of significant connection
                # ground truth matrix using ccorr
                ccorr_combined_ground_truth_matrices = [
                    theta_ccorr,
                    alpha_ccorr,
                    beta_ccorr,
                    gamma_ccorr,
                ]
                # ground truth matrix using plv
                plv_combined_ground_truth_matrices = [
                    theta_plv,
                    alpha_plv,
                    beta_plv,
                    gamma_plv,
                ]
                # ground truth matrix using coh
                coh_combined_ground_truth_matrices = [
                    theta_coh,
                    alpha_coh,
                    beta_coh,
                    gamma_coh,
                ]

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

                # Define lists that contain significant actual scores (ccor, plv, coh) along with electrode pair labels

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
                        epoch1 = load_epoch_S1.get_data(
                            picks=ch_names[participant1_channel]
                        )
                        epoch2 = load_epoch_S2.get_data(
                            picks=ch_names[participant2_channel]
                        )

                        rng = np.random.default_rng(42)  # set a random seed

                        # Permute for several times as defined above
                        n_perms = n_permutations

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
                            epoch2_in_permuted_order = [epoch2[i] for i in perm1]

                            # combine the two permuted epochs together
                            # data_inter_permuted = np.array([epoch1, epoch2])
                            data_inter_permuted = np.array(
                                [epoch1_in_permuted_order, epoch2_in_permuted_order]
                            )

                            # Calculate ccorr or plv or coh of two permuted data. We only need to calculate complex_signal once for all 'ccorr', 'plv' and 'coh' because these all use same complex_signal
                            complex_signal = analyses.compute_freq_bands(
                                data_inter_permuted, sampling_rate, freq_bands
                            )

                            # Computing frequency- and time-frequency-domain connectivity, 'ccorr', 'plv', 'coh'.
                            # add analyses.compute_sync(...) if you would like to test more connectivity analysis method e.g. 'pdc', etc.
                            result_ccorr = analyses.compute_sync(
                                complex_signal, mode="ccorr"
                            )
                            result_plv = analyses.compute_sync(
                                complex_signal, mode="plv"
                            )
                            result_coh = analyses.compute_sync(
                                complex_signal, mode="coh"
                            )

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
                ccorr_combined_freq_n_connections_list = list(
                    ccorr_combined_freq_n_connections
                )
                plv_combined_freq_n_connections_list = list(
                    plv_combined_freq_n_connections
                )
                coh_combined_freq_n_connections_list = list(
                    coh_combined_freq_n_connections
                )

                # Progress getting actual score from marker significant connection matrix (ccorr)
                # Iterate over frequency (there are 4 , ie. theta, alpha, beta, gamma)
                theta_sig_electrode_pair_ccorr = []
                alpha_sig_electrode_pair_ccorr = []
                beta_sig_electrode_pair_ccorr = []
                gamma_sig_electrode_pair_ccorr = []

                for idx_freq in range(len(ccorr_combined_freq_n_connections)):
                    # Iterate over row of matrix (16 x 16)
                    for idx_row, row in enumerate(
                        ccorr_combined_freq_n_connections[idx_freq]
                    ):
                        # Iterate over column of matrix (16 x 16)
                        for idx_col, col in enumerate(
                            ccorr_combined_freq_n_connections[idx_freq][idx_row]
                        ):
                            if (
                                ccorr_combined_freq_n_connections[idx_freq][idx_row][
                                    idx_col
                                ]
                                == 1
                            ):
                                idx_sig_connection = tuple([idx_row, idx_col])
                                # Get actual score
                                actual_score = ccorr_combined_ground_truth_matrices[
                                    idx_freq
                                ][idx_row][idx_col]
                                # Get pair label of electorode
                                sig_electrode_pair_label = (
                                    get_electrode_labels_connections(idx_sig_connection)
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
                    for idx_row, row in enumerate(
                        plv_combined_freq_n_connections[idx_freq]
                    ):
                        # Iterate over column of matrix (16 x 16)
                        for idx_col, col in enumerate(
                            plv_combined_freq_n_connections[idx_freq][idx_row]
                        ):
                            if (
                                plv_combined_freq_n_connections[idx_freq][idx_row][
                                    idx_col
                                ]
                                == 1
                            ):
                                idx_sig_connection = tuple([idx_row, idx_col])
                                # Get actual score
                                actual_score = plv_combined_ground_truth_matrices[
                                    idx_freq
                                ][idx_row][idx_col]
                                # Get pair label of electorode
                                sig_electrode_pair_label = (
                                    get_electrode_labels_connections(idx_sig_connection)
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
                plv_combined_freqs_electrode_pair_n_actual_score.append(
                    beta_sig_electrode_pair_plv
                )
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
                    for idx_row, row in enumerate(
                        coh_combined_freq_n_connections[idx_freq]
                    ):
                        # Iterate over column of matrix (16 x 16)
                        for idx_col, col in enumerate(
                            coh_combined_freq_n_connections[idx_freq][idx_row]
                        ):
                            if (
                                coh_combined_freq_n_connections[idx_freq][idx_row][
                                    idx_col
                                ]
                                == 1
                            ):
                                idx_sig_connection = tuple([idx_row, idx_col])
                                # Get actual score
                                actual_score = coh_combined_ground_truth_matrices[
                                    idx_freq
                                ][idx_row][idx_col]
                                # Get pair label of electorode
                                sig_electrode_pair_label = (
                                    get_electrode_labels_connections(idx_sig_connection)
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
                coh_combined_freqs_electrode_pair_n_actual_score.append(
                    beta_sig_electrode_pair_coh
                )
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

    def count_sig_connections(self, path: str):
        """
            Count a number of significant connections for a certain eye condition, eg. averted_pre.\
            Divided into different algorithms (ccorr, coh, and plv) and frequencies (theta, alpha, beta, and gamma).

            :param path: a path that contains ``*pkl`` file which contains actual scores of connections.                        
            :type path: str
            :return all_connections: it returns multiple values.
            :rtype: namedtuple

            .. note::
                * parameters:
                    * ``path``: 
                    * Each ``*.pkl`` file will have a lenght of 4 (the order is theta, alpha, beta, and gamma).
                    * It is the output of :meth:`permut_sig_connection`.

                * returns:
                    * The order of ``all_connections (namedtuple)`` is described below :
                    * REMEMBER ! Index starts from ``zero`` in python

                    0. total_sig_ccorr_theta_connections,
                    1. total_sig_ccorr_alpha_connections,
                    #. total_sig_ccorr_beta_connections,
                    #. total_sig_ccorr_gamma_connections,
                    #. total_sig_coh_theta_connections,
                    #. total_sig_coh_alpha_connections,
                    #. total_sig_coh_beta_connections,
                    #. total_sig_coh_gamma_connections,
                    #. total_sig_plv_theta_connections,
                    #. total_sig_plv_alpha_connections,
                    #. total_sig_plv_beta_connections,
                    #. total_sig_plv_gamma_connections.
            
            .. seealso::
                :meth:`permut_sig_connection`
                   

        """

        results = namedtuple(
            "results",
            [
                "total_sig_ccorr_theta_connections",
                "total_sig_ccorr_alpha_connections",
                "total_sig_ccorr_beta_connections",
                "total_sig_ccorr_gamma_connections",
                "total_sig_coh_theta_connections",
                "total_sig_coh_alpha_connections",
                "total_sig_coh_beta_connections",
                "total_sig_coh_gamma_connections",
                "total_sig_plv_theta_connections",
                "total_sig_plv_alpha_connections",
                "total_sig_plv_beta_connections",
                "total_sig_plv_gamma_connections",
            ],
        )

        files = os.listdir(path)
        # Create new list to count the number of significant connection (eg. list_at, list_aa, list_ab, list_ag)
        ccorr_sig_connections = []
        coh_sig_connections = []
        plv_sig_connections = []

        # Separate files into different container according to algorithm
        for file in files:
            # ccorr
            if "actual_score_data" in file and "ccorr" in file:
                ccorr_sig_connections.append(file)
                # Sort the list
                ccorr_sig_connections.sort()
            # coh
            elif "actual_score_data" in file and "coh" in file:
                coh_sig_connections.append(file)
                # Sort the list
                coh_sig_connections.sort()
            # plv
            elif "actual_score_data" in file and "plv" in file:
                plv_sig_connections.append(file)
                # Sort the list
                plv_sig_connections.sort()

        # Define list for ccorr per frequency
        total_sig_ccorr_theta_connections = []
        total_sig_ccorr_alpha_connections = []
        total_sig_ccorr_beta_connections = []
        total_sig_ccorr_gamma_connections = []

        # Define list for coh per frequency
        total_sig_coh_theta_connections = []
        total_sig_coh_alpha_connections = []
        total_sig_coh_beta_connections = []
        total_sig_coh_gamma_connections = []

        # Define list for plv per frequency
        total_sig_plv_theta_connections = []
        total_sig_plv_alpha_connections = []
        total_sig_plv_beta_connections = []
        total_sig_plv_gamma_connections = []

        # Count significant connection for ccorr algorithm and separate into 4 frequencies:
        # theta, alpha, beta, and gamma
        for file in ccorr_sig_connections:
            ccorr_file_2_read = os.path.join(path, file)
            ccorr_file = read_pickle(ccorr_file_2_read)

            # Theta = 0th index in the list
            sig_ccorr_theta_connections = len(ccorr_file[0])
            total_sig_ccorr_theta_connections.append(sig_ccorr_theta_connections)

            # Alpha = 1st index in the list
            sig_ccorr_alpha_connections = len(ccorr_file[1])
            total_sig_ccorr_alpha_connections.append(sig_ccorr_alpha_connections)

            # Beta = 2nd index in the list
            sig_ccorr_beta_connections = len(ccorr_file[2])
            total_sig_ccorr_beta_connections.append(sig_ccorr_beta_connections)

            # Gamma = 3rd index in the list
            sig_ccorr_gamma_connections = len(ccorr_file[3])
            total_sig_ccorr_gamma_connections.append(sig_ccorr_gamma_connections)

        # Count significant connection for coh algorithm and separate into 4 frequencies:
        # theta, alpha, beta, and gamma
        for file in coh_sig_connections:
            coh_file_2_read = os.path.join(path, file)
            coh_file = read_pickle(coh_file_2_read)

            # Theta = 0th index in the list
            sig_coh_theta_connections = len(coh_file[0])
            total_sig_coh_theta_connections.append(sig_coh_theta_connections)

            # Alpha = 1st index in the list
            sig_coh_alpha_connections = len(coh_file[1])
            total_sig_coh_alpha_connections.append(sig_coh_alpha_connections)

            # Beta = 2nd index in the list
            sig_coh_beta_connections = len(coh_file[2])
            total_sig_coh_beta_connections.append(sig_coh_beta_connections)

            # Gamma = 3rd index in the list
            sig_coh_gamma_connections = len(coh_file[3])
            total_sig_coh_gamma_connections.append(sig_coh_gamma_connections)

        # Count significant connection for plv algorithm and separate into 4 frequencies:
        # theta, alpha, beta, and gamma
        for file in plv_sig_connections:
            plv_file_2_read = os.path.join(path, file)
            plv_file = read_pickle(plv_file_2_read)

            # Theta = 0th index in the list
            sig_plv_theta_connections = len(plv_file[0])
            total_sig_plv_theta_connections.append(sig_plv_theta_connections)

            # Alpha = 1st index in the list
            sig_plv_alpha_connections = len(plv_file[1])
            total_sig_plv_alpha_connections.append(sig_plv_alpha_connections)

            # Beta = 2nd index in the list
            sig_plv_beta_connections = len(plv_file[2])
            total_sig_plv_beta_connections.append(sig_plv_beta_connections)

            # Gamma = 3rd index in the list
            sig_plv_gamma_connections = len(plv_file[3])
            total_sig_plv_gamma_connections.append(sig_plv_gamma_connections)

        all_connections = results(
            total_sig_ccorr_theta_connections,
            total_sig_ccorr_alpha_connections,
            total_sig_ccorr_beta_connections,
            total_sig_ccorr_gamma_connections,
            total_sig_coh_theta_connections,
            total_sig_coh_alpha_connections,
            total_sig_coh_beta_connections,
            total_sig_coh_gamma_connections,
            total_sig_plv_theta_connections,
            total_sig_plv_alpha_connections,
            total_sig_plv_beta_connections,
            total_sig_plv_gamma_connections,
        )

        return all_connections

    def diff_n_connections_pre_post(
        self,
        averted_pre: tuple,
        averted_post: tuple,
        direct_pre: tuple,
        direct_post: tuple,
        natural_pre: tuple,
        natural_post: tuple,
    ):
        """
        Objective  : To find difference (absolute number) between pre and post for each eye condition, combination algorithm and frequency

        Parameters :

                     These are the results of count_sig_connections function. Run it for each eye condition
                     each result will become an input of this function.

        Outputs    :
                     - diff_averted
                     - diff_direct
                     - diff_natural

                    NOTE : Read the notes below to understand the structure of the above output of three variables

                    These are the order of list for each eye condition (diff_averted, diff_direct, diff_natural)
                    total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
                    total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
                    total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections

        """

        diff_averted = []
        diff_direct = []
        diff_natural = []

        for i in range(
            len(averted_pre)
        ):  # NOTE : The length is 12 means there are 12 outputs
            # that are resulted from the count_sig_connections function
            # Just pick up averted_pre variable
            diff_averted.append(
                [np.abs(x - y) for x, y in zip(averted_post[i], averted_pre[i])]
            )

            diff_direct.append(
                [np.abs(x - y) for x, y in zip(direct_post[i], direct_pre[i])]
            )

            diff_natural.append(
                [np.abs(x - y) for x, y in zip(natural_post[i], natural_pre[i])]
            )

        return diff_averted, diff_direct, diff_natural

    def corr_eeg_connection_n_question(
        self, diff_connection: List[list], diff_scale: list, title: str
    ):
        """
        Objective  : Analyze pearson correlation between number of connections of EEG
                    (substracted between post and pre) and subscale of SPGQ or SPGQ total score

        Parameters :
                    - diff_connection(List[list]) : Substracted number of connections of EEG. Each list will have six order
                                                    as follow :  Resulted from EEG.Analysis.diff_n_connections_pre_post funct

                                                diff_connect_ccorr_theta_connections, diff_connect_ccorr_alpha_connections, diff_connect_ccorr_beta_connections, diff_connect_ccorr_gamma_connections,
                                                diff_connect_coh_theta_connections, diff_connect_coh_alpha_connections, diff_connect_coh_beta_connections, diff_connect_coh_gamma_connections,
                                                diff_connect_plv_theta_connections, diff_connect_plv_alpha_connections, diff_connect_plv_beta_connections, diff_connect_plv_gamma_connections


                    - diff_scale(list) :  Substracted subscale / total score of SPGQ between pre and post

                                            - "Empathy SPGQ"
                                            - "NegativeFeelings SPGQ"
                                            - "Behavioural SPGQ"
                                            - "SPGQ Total"
                                            - "CoPresence Total"
                                            Resulted from Questionnnaire.questionnaire.diff_score_questionnaire_pre_post funct

                    - title (str)      : Title of correlation between which eye condition and subscale of questionnaire

        Output     :
                     Print Correlational score between the following connections and subscale of questionnaire (SPGQ)

                     diff_connect_ccorr_theta_connections, diff_connect_ccorr_alpha_connections, diff_connect_ccorr_beta_connections, diff_connect_ccorr_gamma_connections,
                     diff_connect_coh_theta_connections, diff_connect_coh_alpha_connections, diff_connect_coh_beta_connections, diff_connect_coh_gamma_connections,
                     diff_connect_plv_theta_connections, diff_connect_plv_alpha_connections, diff_connect_plv_beta_connections, diff_connect_plv_gamma_connections

        """

        print(title)
        for i in range(len(diff_connection)):
            print(f"{i}, {pearsonr(diff_connection[i], diff_scale)}")

    def corr_eeg_connection_n_eye(
        self, diff_eye_cond: list, diff_eeg_connections: List[list]
    ):
        """
        Objective : Find correlation between eye gaze percentage of looking and a number of eeg connections

        Parameters:
                    - diff_eye_cond (list)              : Percentage of looking that has been deducted between pre and post
                    - diff_eeg_connections (List[list]) : Number of connections of EEG that has been deducted between pre and post.
                                                          It is the output of diff_n_connections_pre_pos function

        Output    :
                    corr_eye_eeg(list): Pearson correlational values. This is the order :

                                        total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
                                        total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
                                        total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections
        """
        corr_eye_eeg = []
        for idx, value in enumerate(diff_eeg_connections):
            corr_eye_eeg.append(pearsonr(diff_eye_cond, diff_eeg_connections[idx]))
        return corr_eye_eeg

    def plot_eeg_connection_n_question(
        self,
        x_axis_diff_connection: list,
        y_axis_diff_scale: list,
        title: str,
        xlabel: str,
        ylabel: str,
    ):
        """
        Objective : Plot a correlation (scatter plot) between number of connections (EEG) and
                    score of subscale of SPGQ / Co-Presence

        Parameters :
                    - x_axis_diff_connection (list) : (data for x axis) Number of connections for a certain eye conditon, algorithm, and frequency
                    - y_axis_diff_scale (list)      : (data for y axis) Score of subscale for a certain eye conditon
                                                        - "Empathy SPGQ"
                                                        - "NegativeFeelings SPGQ"
                                                        - "Behavioural SPGQ"
                                                        - "SPGQ Total"
                                                        - "CoPresence Total"
                                                        Take ONE of the lists that is resulted from EEG.stats.diff_n_connections_pre_post funct
                                                        as an input

                    - title (str)                   : Title for the plot
                    - xlabel (str)                  : Xlabel for the plot
                    - ylabel (str)                  : Ylabel for the plot

        Output     :
                      Plot
        """

        # adds the title
        plt.title(title)

        # plot the data
        plt.scatter(x_axis_diff_connection, y_axis_diff_scale)

        # fits the best fitting line to the data
        plt.plot(
            np.unique(x_axis_diff_connection),
            np.poly1d(np.polyfit(x_axis_diff_connection, y_axis_diff_scale, 1))(
                np.unique(x_axis_diff_connection)
            ),
            color="red",
        )

        # Labelling axes
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    
    def calculate_eeg_spgq_total_correlations(self, averted_pre_actual_score_connections, averted_post_actual_score_connections,
                                                direct_pre_actual_score_connections, direct_post_actual_score_connections,
                                                natural_pre_actual_score_connections, natural_post_actual_score_connections,
                                                questionnaire_path):
        """
        Calculate Pearson correlational scores between EEG connections and SPGQ Total scores for different eye conditions.

        Args:
        averted_pre_actual_score_connections (str): Path to actual scores of EEG connections for Averted pre condition.
        averted_post_actual_score_connections (str): Path to actual scores of EEG connections for Averted post condition.
        direct_pre_actual_score_connections (str): Path to actual scores of EEG connections for Direct pre condition.
        direct_post_actual_score_connections (str): Path to actual scores of EEG connections for Direct post condition.
        natural_pre_actual_score_connections (str): Path to actual scores of EEG connections for Natural pre condition.
        natural_post_actual_score_connections (str): Path to actual scores of EEG connections for Natural post condition.
        questionnaire_path (str): Path to SPGQ questionnaire data.

        Returns:
        tuple: Tuple containing lists of correlational scores for Averted, Direct, and Natural conditions.
            Each list contains tuples with frequency band, correlational score, and p-value marked with '*' if p-value is less than 0.05.
        """
        # EEG
        
        # Calculate the number of connections (EEG)
        averted_pre_sig_connect = self.count_sig_connections(averted_pre_actual_score_connections)
        averted_post_sig_connect = self.count_sig_connections(averted_post_actual_score_connections)
        direct_pre_sig_connect = self.count_sig_connections(direct_pre_actual_score_connections)
        direct_post_sig_connect = self.count_sig_connections(direct_post_actual_score_connections)
        natural_pre_sig_connect = self.count_sig_connections(natural_pre_actual_score_connections)
        natural_post_sig_connect = self.count_sig_connections(natural_post_actual_score_connections)

        # Calculate the difference between post and pre (EEG connections)
        diff_averted, diff_direct, diff_natural = self.diff_n_connections_pre_post(
            averted_pre_sig_connect, averted_post_sig_connect,
            direct_pre_sig_connect, direct_post_sig_connect,
            natural_pre_sig_connect, natural_post_sig_connect
        )

        # Extract ccor algorithm only
        diff_averted = diff_averted[:4]  # The order is theta, alpha, beta, and gamma
        diff_direct = diff_direct[:4]  # The order is theta, alpha, beta, and gamma
        diff_natural = diff_natural[:4]  # The order is theta, alpha, beta, and gamma

        # SPGQ
        questionnaire = Questionnaire()

        # Scoring questionnaire (SPGQ)
        all_questionnaires_scoring = questionnaire.scoring_questionnaire(questionnaire_path)

        # Calculate the difference of SPGQ and its subscales between post and pre (SPGQ)
        all_questionnaires_scoring_diff_empathy = questionnaire.diff_score_questionnaire_pre_post(
            all_questionnaires_scoring[0], all_questionnaires_scoring[1],
            all_questionnaires_scoring[2], all_questionnaires_scoring[3],
            all_questionnaires_scoring[4], all_questionnaires_scoring[5],
            "Empathy SPGQ"
        )

        all_questionnaires_scoring_diff_neg_feeling = questionnaire.diff_score_questionnaire_pre_post(
            all_questionnaires_scoring[0], all_questionnaires_scoring[1],
            all_questionnaires_scoring[2], all_questionnaires_scoring[3],
            all_questionnaires_scoring[4], all_questionnaires_scoring[5],
            "NegativeFeelings SPGQ"
        )

        all_questionnaires_scoring_diff_behav = questionnaire.diff_score_questionnaire_pre_post(
            all_questionnaires_scoring[0], all_questionnaires_scoring[1],
            all_questionnaires_scoring[2], all_questionnaires_scoring[3],
            all_questionnaires_scoring[4], all_questionnaires_scoring[5],
            "Behavioural SPGQ"
        )

        all_questionnaires_scoring_diff_spg_total = questionnaire.diff_score_questionnaire_pre_post(
            all_questionnaires_scoring[0], all_questionnaires_scoring[1],
            all_questionnaires_scoring[2], all_questionnaires_scoring[3],
            all_questionnaires_scoring[4], all_questionnaires_scoring[5],
            "SPGQ Total"
        )

        # Correlational score calculation
        print("Averted-SPGQ-Total")
        freqs = ["theta", "alpha", "beta", "gamma"]
        averted_spgq_total_correlations = []

        for i in range(len(diff_averted)):  # Iterate over the freq (theta, alpha, beta, and gamma)
            correlation_result = pearsonr(diff_averted[i], list(all_questionnaires_scoring_diff_spg_total[0]))
            correlational_score = correlation_result[0]
            p_value = correlation_result[1]
            if p_value < 0.05:
                correlational_score_str = f"{correlational_score} * ({p_value})"
            else:
                correlational_score_str = f"{correlational_score} ({p_value})"
            averted_spgq_total_correlations.append((freqs[i], correlational_score_str))

        print("Direct-SPGQ-Total")
        direct_spgq_total_correlations = []

        for i in range(len(diff_direct)):  # Iterate over the freq (theta, alpha, beta, and gamma)
            correlation_result = pearsonr(diff_direct[i], list(all_questionnaires_scoring_diff_spg_total[1]))
            correlational_score = correlation_result[0]
            p_value = correlation_result[1]
            if p_value < 0.05:
                correlational_score_str = f"{correlational_score} * ({p_value})"
            else:
                correlational_score_str = f"{correlational_score} ({p_value})"
            direct_spgq_total_correlations.append((freqs[i], correlational_score_str))

        print("Natural-SPGQ-Total")
        natural_spgq_total_correlations = []

        for i in range(len(diff_natural)):  # Iterate over the freq (theta, alpha, beta, and gamma)
            correlation_result = pearsonr(diff_natural[i], list(all_questionnaires_scoring_diff_spg_total[2]))
            correlational_score = correlation_result[0]
            p_value = correlation_result[1]
            if p_value < 0.05:
                correlational_score_str = f"{correlational_score} * ({p_value})"
            else:
                correlational_score_str = f"{correlational_score} ({p_value})"
            natural_spgq_total_correlations.append((freqs[i], correlational_score_str))

        # Return the three lists of correlational scores
        return (
            averted_spgq_total_correlations,
            direct_spgq_total_correlations,
            natural_spgq_total_correlations
        )

