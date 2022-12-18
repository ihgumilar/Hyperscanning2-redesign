# %% Relevant packages
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
from hypyp import analyses, prep, stats, viz
from hypyp.ext.mpl3d import glm
from hypyp.ext.mpl3d.camera import Camera
from hypyp.ext.mpl3d.mesh import Mesh
from icecream import ic
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %%
class Time_Conversion:
    """
    Low-level module
    """

    # Time conversion
    def convert(self, seconds: int or float):
        """Convert elapsed time into hour, minutes, seconds

        :param seconds: number of elapsed seconds
        :type seconds: int or float
        :return: hour:minutes:seconds
        :rtype: str
        """

        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return "%d:%02d:%02d" % (hour, minutes, seconds)


class _Intermediary_Clean_EEG:
    """
    Abstract-module
    """

    def __init__(self):
        self._convert = Time_Conversion()

    def convert(self, seconds: int or float):
        hour_min_sec = self._convert.convert(seconds)

        return hour_min_sec


class Compute_Sync_EEG_Exp2_Redesign:
    """Initialization of using :literal:`Compute_Sync_EEG_Exp2_Redesign` class.

    High-level module to compute inter-brain synchrony and clean up EEG data simultaneously.

        :param n_raw_files: number of raw EEG files (*.csv)
        :type n_raw_files: int
        :param algorithm: algorithm for calculating inter-brain synchrony, defaults to "ccorr"
        :type algorithm: str, optional
        :param bad_channels: bad channels, defaults to [ [], ["Fp2", "C3"], [], ["Fp2", "C3"], [], ["Fp2", "C3"], [], ["Fp2", "F8", "C3"], [], ["Fp2", "C3"], [], ["Fp2", "C3"], [], ["Fp2", "C3"], [], [], [], [], ["F7"], ["Fp2", "T7"], [], [], [], ["Fp2", "F4", "C3"], [], ["Fp2", "F4", "C3", "C4"], [], [], [], [], ]
        :type bad_channels: list, optional

    .. note::
        Supported connectivity measures (algorithm) :
            * ``"envelope_corr"``: envelope correlation
            * ``"pow_corr"``: power correlation
            * ``"plv"``: phase locking value
            * ``"ccorr"``: circular correlation coefficient
            * ``"coh"``: coherence
            * ``imaginary_coh``: imaginary coherence
            * ``"pli"`` : phase lag index
            * ``"wpli"`` : weighted phase lag index

    .. seealso::
        For more updated supported connectivity measures (algorithm) in `HyPyP module. <https://hypyp.readthedocs.io/en/latest/API/analyses/#hypyp.analyses.compute_sync>`_


    """

    def __init__(
        self,
        n_raw_files: int,
        algorithm: str = "ccorr",
        bad_channels=[
            # S1
            [],
            # S2
            ["Fp2", "C3"],
            # S3
            [],
            # S4
            ["Fp2", "C3"],
            # S5
            [],
            # S6
            ["Fp2", "C3"],
            # S7
            [],
            # S8
            ["Fp2", "F8", "C3"],
            # S9
            [],
            # S10
            ["Fp2", "C3"],
            # S11
            [],
            # S12
            ["Fp2", "C3"],
            # S13
            [],
            # S14
            ["Fp2", "C3"],
            # S15
            [],
            # S16
            [],
            # S17
            [],
            # S18
            [],
            # S19
            ["F7"],
            # S20
            ["Fp2", "T7"],
            # S21
            [],
            # S22
            [],
            # S23
            [],
            # S24
            ["Fp2", "F4", "C3"],
            # S25
            [],
            # S26
            ["Fp2", "F4", "C3", "C4"],
            # S27
            [],
            # S28
            [],
            # S29
            [],
            # S30
            [],
        ],
    ):

        self.number_raw_files = n_raw_files
        self.__algorithm = algorithm
        self._bad_channels = bad_channels
        self.__intermediary_clean_eeg = _Intermediary_Clean_EEG()

    def clean_N_compute_sync_eeg_averted_pre_exp(
        self,
        experimental_data: str,
        preprocessed_data: str,
        sync_score_n_del_indices: str,
    ):
        """
        #. Clean up noises of EEG files that have been combined
        #. Delete bad epochs
        #. Calculate score of inter-brain synchrony


        :param experimental_data: folder where the combined files (averted_pre) of experiment that will be cleaned are stored(Subject 1 and 2).
        :type experimental_data: str
        :param preprocessed_data: folder to which will store the pre-processed epoched data(*.fif).
        :type preprocessed_data: str
        :param sync_score_n_del_indices: folder to which will store the indices of deleted epoch of EEG data.
        :type sync_score_n_del_indices: str
        :returns:
            #. Pre-processed epoched EEG file of subject 1 (*.fif)
            #. Pre-processed epoched EEG file of subject 2 (*.fif)
            #. Scores of inter-brain synchrony (*.pkl files)
        :rtype: ``*.fif`` (mne) and ``*.pkl``

        .. note::
            returns:
                #. Pre-processed epoched EEG file of subject 1 (``*.fif``).
                #. Pre-processed epoched EEG file of subject 2 (``*.fif``).
                #. Scores of inter-brain synchrony (*.pkl files).

                    * The structure of pkl files is each pair will have 4 lists, which has the following order.
                    * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,\
                       then move the 2nd four lists which belong to pair 2 and so on.    
                #. Indices of deleted epochs of EEG (``*.pkl`` files)

                    * Length of list, once pkl file is loaded, is equal to the number of pairs.
                    * For instance, if we have 13 pairs, then there will be 13 lists within that ``*.pkl`` file.
        """

        odd_subject_averted_pre_suffix = (
            "-averted_pre_right_left_point_combined_raw.fif"
        )
        even_subject_averted_pre_suffix = (
            "-averted_pre_left_right_point_combined_raw.fif"
        )

        start = timer()

        all_deleted_epochs_indices_averted_pre = []

        total_n_connections_all_pairs_averted_pre = []

        list_scores_all_theta = []
        list_scores_all_alpha = []
        list_scores_all_beta = []
        list_scores_all_gamma = []
        list_scores_all_averted_pre = []

        total_n_connections_theta = []
        total_n_connections_alpha = []
        total_n_connections_beta = []
        total_n_connections_gamma = []
        total_n_connections_all_pairs_averted_pre = []

        # Note : no_filter = No statistical analysis yet conducted to see which connections that are significant
        list_averted_pre_no_filter_all = []

        original_bad_channels_all = self._bad_channels

        # To loop subject number.
        # Every loop there are two files that are taken (odd-even subject)

        begin = 0
        end = self.number_raw_files
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
        ):

            os.chdir(experimental_data)

            # If we want to exclude pair that wants to be skipped
            # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
            # Uncomment this line, in case you have files that want to be skipped
            # if (i == 2):  # NOTE: Indicate pair
            #     continue

            # Subject no. 1 - 10
            if (i + 1) <= 9:
                fname1_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 1)
                    + odd_subject_averted_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 2)
                    + even_subject_averted_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 10:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_averted_pre_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 11 - 20
            elif (i + 1) >= 11:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_averted_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_averted_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 20:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_averted_pre_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 21 - 30
            elif (i + 1) >= 21:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_averted_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_averted_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 30:
                    fname2_direct = (
                        experimental_data
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

            raw1_direct = mne.io.read_raw_fif(
                fname_S1_direct, preload=True, verbose=False
            )
            raw2_direct = mne.io.read_raw_fif(
                fname_S2_direct, preload=True, verbose=False
            )

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
            (
                cleaned_epochs_AR,
                percentage_rejected_epoch,
                delete_epochs_indices,
            ) = prep.AR_local(cleaned_epochs_ICA, verbose=True)

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
                preproc_S1,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            psd2 = analyses.pow(
                preproc_S2,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            data_psd = np.array([psd1.psd, psd2.psd])

            #
            # # Connectivity
            # with ICA
            data_inter = np.array([preproc_S1, preproc_S2])
            # Computing analytic signal per frequency band
            # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
            complex_signal = analyses.compute_freq_bands(
                data_inter, sampling_rate, freq_bands
            )
            # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
            result = analyses.compute_sync(complex_signal, mode=self.__algorithm)
            #
            # NOTE: Slicing results to get the Inter-brain part of the matrix.
            # Refer to this for slicing or counting no. of connections later on
            #
            n_ch = len(preproc_S1.info["ch_names"])
            theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

            list_averted_pre_no_filter_all.append(theta)
            list_averted_pre_no_filter_all.append(alpha)
            list_averted_pre_no_filter_all.append(beta)
            list_averted_pre_no_filter_all.append(gamma)

            # Check if inter-brain connection scores have been put into a list
            print(
                f"(averted_pre) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
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
            os.chdir(preprocessed_data)

            # Save pre-processed (epoched) data of subject 1
            preproc_S1.save(epoched_file_name_S1, overwrite=True)

            # Save pre-processed (epoched) data of subject 2
            preproc_S2.save(epoched_file_name_S2, overwrite=True)

        # Change to a directory where we want to save the above populated lists (pre-processed data)
        os.chdir(sync_score_n_del_indices)

        # Save the scores of inter-brain synchrony from each pair into pkl file
        filename_sync = (
            "List_" + self.__algorithm + "_scores_all_pairs_averted_pre_no_filter.pkl"
        )

        with open(filename_sync, "wb") as handle:
            pickle.dump(
                list_averted_pre_no_filter_all,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # NOTE : The structure of files is each pair will have 4 lists, which has the following order
        #        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
        #        * then move the 2nd four lists which belong to pair 2 and so on.
        print(
            "(averted_pre) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
        )

        # Save indices of deleted epochs from each pair into pkl file
        # NOTE : Length of list once pkl file is loaded is equal to the number of pairs
        # If we have 15 pairs, then there will be 15 lists within that pkl file
        with open("list_deleted_epoch_indices_averted_pre.pkl", "wb") as handle:
            pickle.dump(
                all_deleted_epochs_indices_averted_pre,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(
            "(averted_pre) All indices of deleted epochs have been saved into a pickle file"
        )

        # Count elapsed time
        end = timer()
        # Calling function to convert seconds to hour minute, second
        print(f"Processed time : {self.__intermediary_clean_eeg.convert(end - start)}")  # type: ignore

    def clean_N_compute_sync_eeg_averted_post_exp(
        self,
        experimental_data: str,
        preprocessed_data: str,
        sync_score_n_del_indices: str,
    ):
        """
        #. Clean up noises of EEG files that have been combined
        #. Delete bad epochs
        #. Calculate score of inter-brain synchrony


        :param experimental_data: folder where the combined files (averted_post) of experiment that will be cleaned are stored(Subject 1 and 2).
        :type experimental_data: str
        :param preprocessed_data: folder to which will store the pre-processed epoched data(*.fif).
        :type preprocessed_data: str
        :param sync_score_n_del_indices: folder to which will store the indices of deleted epoch of EEG data.
        :type sync_score_n_del_indices: str
        :returns:
            #. Pre-processed epoched EEG file of subject 1 (*.fif)
            #. Pre-processed epoched EEG file of subject 2 (*.fif)
            #. Scores of inter-brain synchrony (*.pkl files)
        :rtype: ``*.fif`` (mne) and ``*.pkl``

        .. note::
            returns:
                #. Pre-processed epoched EEG file of subject 1 (``*.fif``).
                #. Pre-processed epoched EEG file of subject 2 (``*.fif``).
                #. Scores of inter-brain synchrony (*.pkl files).

                    * The structure of pkl files is each pair will have 4 lists, which has the following order.
                    * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,\
                       then move the 2nd four lists which belong to pair 2 and so on.    
                #. Indices of deleted epochs of EEG (``*.pkl`` files)

                    * Length of list, once pkl file is loaded, is equal to the number of pairs.
                    * For instance, if we have 13 pairs, then there will be 13 lists within that ``*.pkl`` file.
        """

        odd_subject_averted_post_suffix = (
            "-averted_post_right_left_point_combined_raw.fif"
        )
        even_subject_averted_post_suffix = (
            "-averted_post_left_right_point_combined_raw.fif"
        )

        start = timer()

        all_deleted_epochs_indices_averted_post = []

        total_n_connections_all_pairs_averted_post = []

        list_scores_all_theta = []
        list_scores_all_alpha = []
        list_scores_all_beta = []
        list_scores_all_gamma = []
        list_scores_all_averted_post = []

        total_n_connections_theta = []
        total_n_connections_alpha = []
        total_n_connections_beta = []
        total_n_connections_gamma = []
        total_n_connections_all_pairs_averted_post = []

        # Note : no_filter = No statistical analysis yet conducted to see which connections that are significant
        list_averted_post_no_filter_all = []

        original_bad_channels_all = self._bad_channels

        # To loop subject number.
        # Every loop there are two files that are taken (odd-even subject)

        begin = 0
        end = self.number_raw_files
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
        ):

            os.chdir(experimental_data)

            # If we want to exclude pair that wants to be skipped
            # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
            # Uncomment this line, in case you have files that want to be skipped
            # if (i == 2):  # NOTE: Indicate pair
            #     continue

            # Subject no. 1 - 10
            if (i + 1) <= 9:
                fname1_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 1)
                    + odd_subject_averted_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 2)
                    + even_subject_averted_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 10:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_averted_post_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 11 - 20
            elif (i + 1) >= 11:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_averted_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_averted_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 20:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_averted_post_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 21 - 30
            elif (i + 1) >= 21:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_averted_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_averted_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 30:
                    fname2_direct = (
                        experimental_data
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

            fname_S1_direct = fname1_direct
            fname_S2_direct = fname2_direct

            # Get original bad channels for odd and even subject
            original_bad_channels1 = original_bad_channels_all[i]
            original_bad_channels2 = original_bad_channels_all[i + 1]

            raw1_direct = mne.io.read_raw_fif(
                fname_S1_direct, preload=True, verbose=False
            )
            raw2_direct = mne.io.read_raw_fif(
                fname_S2_direct, preload=True, verbose=False
            )

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
            (
                cleaned_epochs_AR,
                percentage_rejected_epoch,
                delete_epochs_indices,
            ) = prep.AR_local(cleaned_epochs_ICA, verbose=True)

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
                preproc_S1,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            psd2 = analyses.pow(
                preproc_S2,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            data_psd = np.array([psd1.psd, psd2.psd])

            #
            # # Connectivity
            # with ICA
            data_inter = np.array([preproc_S1, preproc_S2])
            # Computing analytic signal per frequency band
            # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
            complex_signal = analyses.compute_freq_bands(
                data_inter, sampling_rate, freq_bands
            )
            # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
            result = analyses.compute_sync(complex_signal, mode=self.__algorithm)
            #
            # NOTE: Slicing results to get the Inter-brain part of the matrix.
            # Refer to this for slicing or counting no. of connections later on
            #
            n_ch = len(preproc_S1.info["ch_names"])
            theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

            list_averted_post_no_filter_all.append(theta)
            list_averted_post_no_filter_all.append(alpha)
            list_averted_post_no_filter_all.append(beta)
            list_averted_post_no_filter_all.append(gamma)

            # Check if inter-brain connection scores have been put into a list
            print(
                f"(averted_post) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
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
            os.chdir(preprocessed_data)

            # Save pre-processed (epoched) data of subject 1
            preproc_S1.save(epoched_file_name_S1, overwrite=True)

            # Save pre-processed (epoched) data of subject 2
            preproc_S2.save(epoched_file_name_S2, overwrite=True)

        # Change to a directory where we want to save the above populated lists (pre-processed data)
        os.chdir(sync_score_n_del_indices)

        # Save the scores of inter-brain synchrony from each pair into pkl file
        filename_sync = (
            "List_" + self.__algorithm + "_scores_all_pairs_averted_post_no_filter.pkl"
        )

        with open(filename_sync, "wb") as handle:
            pickle.dump(
                list_averted_post_no_filter_all,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # NOTE : The structure of files is each pair will have 4 lists, which has the following order
        #        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
        #        * then move the 2nd four lists which belong to pair 2 and so on.
        print(
            "(averted_post) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
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
        print(
            "(averted_post) All indices of deleted epochs have been saved into a pickle file"
        )

        # Count elapsed time
        end = timer()
        # Calling function to convert seconds to hour minute, second
        print(f"Processed time : {self.__intermediary_clean_eeg.convert(end - start)}")  # type: ignore

    def clean_N_compute_sync_eeg_direct_pre_exp(
        self,
        experimental_data: str,
        preprocessed_data: str,
        sync_score_n_del_indices: str,
    ):
        """
        #. Clean up noises of EEG files that have been combined
        #. Delete bad epochs
        #. Calculate score of inter-brain synchrony


        :param experimental_data: folder where the combined files (direct_pre) of experiment that will be cleaned are stored(Subject 1 and 2).
        :type experimental_data: str
        :param preprocessed_data: folder to which will store the pre-processed epoched data(*.fif).
        :type preprocessed_data: str
        :param sync_score_n_del_indices: folder to which will store the indices of deleted epoch of EEG data.
        :type sync_score_n_del_indices: str
        :returns:
            #. Pre-processed epoched EEG file of subject 1 (*.fif)
            #. Pre-processed epoched EEG file of subject 2 (*.fif)
            #. Scores of inter-brain synchrony (*.pkl files)
        :rtype: ``*.fif`` (mne) and ``*.pkl``

        .. note::
            returns:
                #. Pre-processed epoched EEG file of subject 1 (``*.fif``).
                #. Pre-processed epoched EEG file of subject 2 (``*.fif``).
                #. Scores of inter-brain synchrony (*.pkl files).

                    * The structure of pkl files is each pair will have 4 lists, which has the following order.
                    * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,\
                       then move the 2nd four lists which belong to pair 2 and so on.    
                #. Indices of deleted epochs of EEG (``*.pkl`` files)

                    * Length of list, once pkl file is loaded, is equal to the number of pairs.
                    * For instance, if we have 13 pairs, then there will be 13 lists within that ``*.pkl`` file.
        """

        odd_subject_direct_pre_suffix = "-direct_pre_right_left_point_combined_raw.fif"
        even_subject_direct_pre_suffix = "-direct_pre_left_right_point_combined_raw.fif"

        start = timer()

        all_deleted_epochs_indices_direct_pre = []

        total_n_connections_all_pairs_direct_pre = []

        list_scores_all_theta = []
        list_scores_all_alpha = []
        list_scores_all_beta = []
        list_scores_all_gamma = []
        list_scores_all_direct_pre = []

        total_n_connections_theta = []
        total_n_connections_alpha = []
        total_n_connections_beta = []
        total_n_connections_gamma = []
        total_n_connections_all_pairs_direct_pre = []

        # Note : no_filter = No statistical analysis yet conducted to see which connections that are significant
        list_direct_pre_no_filter_all = []

        original_bad_channels_all = self._bad_channels

        # To loop subject number.
        # Every loop there are two files that are taken (odd-even subject)

        begin = 0
        end = self.number_raw_files
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
        ):

            os.chdir(experimental_data)

            # If we want to exclude pair that wants to be skipped
            # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
            # Uncomment this line, in case you have files that want to be skipped
            # if (i == 2):  # NOTE: Indicate pair
            #     continue

            # Subject no. 1 - 10
            if (i + 1) <= 9:
                fname1_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 1)
                    + odd_subject_direct_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 2)
                    + even_subject_direct_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 10:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_direct_pre_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 11 - 20
            elif (i + 1) >= 11:
                fname1_direct = (
                    experimental_data + "S" + str(i + 1) + odd_subject_direct_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_direct_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 20:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_direct_pre_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 21 - 30
            elif (i + 1) >= 21:
                fname1_direct = (
                    experimental_data + "S" + str(i + 1) + odd_subject_direct_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_direct_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 30:
                    fname2_direct = (
                        experimental_data
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

            raw1_direct = mne.io.read_raw_fif(
                fname_S1_direct, preload=True, verbose=False
            )
            raw2_direct = mne.io.read_raw_fif(
                fname_S2_direct, preload=True, verbose=False
            )

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
            (
                cleaned_epochs_AR,
                percentage_rejected_epoch,
                delete_epochs_indices,
            ) = prep.AR_local(cleaned_epochs_ICA, verbose=True)

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
                preproc_S1,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            psd2 = analyses.pow(
                preproc_S2,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            data_psd = np.array([psd1.psd, psd2.psd])

            #
            # # Connectivity
            # with ICA
            data_inter = np.array([preproc_S1, preproc_S2])
            # Computing analytic signal per frequency band
            # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
            complex_signal = analyses.compute_freq_bands(
                data_inter, sampling_rate, freq_bands
            )
            # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
            result = analyses.compute_sync(complex_signal, mode=self.__algorithm)
            #
            # NOTE: Slicing results to get the Inter-brain part of the matrix.
            # Refer to this for slicing or counting no. of connections later on
            #
            n_ch = len(preproc_S1.info["ch_names"])
            theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

            list_direct_pre_no_filter_all.append(theta)
            list_direct_pre_no_filter_all.append(alpha)
            list_direct_pre_no_filter_all.append(beta)
            list_direct_pre_no_filter_all.append(gamma)

            # Check if inter-brain connection scores have been put into a list
            print(
                f"(direct_pre) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
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
            os.chdir(preprocessed_data)

            # Save pre-processed (epoched) data of subject 1
            preproc_S1.save(epoched_file_name_S1, overwrite=True)

            # Save pre-processed (epoched) data of subject 2
            preproc_S2.save(epoched_file_name_S2, overwrite=True)

        # Change to a directory where we want to save the above populated lists (pre-processed data)
        os.chdir(sync_score_n_del_indices)

        # Save the scores of inter-brain synchrony from each pair into pkl file
        filename_sync = (
            "List_" + self.__algorithm + "_scores_all_pairs_direct_pre_no_filter.pkl"
        )

        with open(filename_sync, "wb") as handle:
            pickle.dump(
                list_direct_pre_no_filter_all,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # NOTE : The structure of files is each pair will have 4 lists, which has the following order
        #        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
        #        * then move the 2nd four lists which belong to pair 2 and so on.
        print(
            "(direct_pre) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
        )

        # Save indices of deleted epochs from each pair into pkl file
        # NOTE : Length of list once pkl file is loaded is equal to the number of pairs
        # If we have 15 pairs, then there will be 15 lists within that pkl file
        with open("list_deleted_epoch_indices_direct_pre.pkl", "wb") as handle:
            pickle.dump(
                all_deleted_epochs_indices_direct_pre,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(
            "(direct_pre) All indices of deleted epochs have been saved into a pickle file"
        )

        # Count elapsed time
        end = timer()
        # Calling function to convert seconds to hour minute, second
        print(f"Processed time : {self.__intermediary_clean_eeg.convert(end - start)}")  # type: ignore

    def clean_N_compute_sync_eeg_direct_post_exp(
        self,
        experimental_data: str,
        preprocessed_data: str,
        sync_score_n_del_indices: str,
    ):
        """
            #. Clean up noises of EEG files that have been combined
            #. Delete bad epochs
            #. Calculate score of inter-brain synchrony


            :param experimental_data: folder where the combined files (direct_post) of experiment that will be cleaned are stored(Subject 1 and 2).
            :type experimental_data: str
            :param preprocessed_data: folder to which will store the pre-processed epoched data(*.fif).
            :type preprocessed_data: str
            :param sync_score_n_del_indices: folder to which will store the indices of deleted epoch of EEG data.
            :type sync_score_n_del_indices: str
            :returns:
                #. Pre-processed epoched EEG file of subject 1 (*.fif)
                #. Pre-processed epoched EEG file of subject 2 (*.fif)
                #. Scores of inter-brain synchrony (*.pkl files)
            :rtype: ``*.fif`` (mne) and ``*.pkl``

            .. note::
                returns:
                    #. Pre-processed epoched EEG file of subject 1 (``*.fif``).
                    #. Pre-processed epoched EEG file of subject 2 (``*.fif``).
                    #. Scores of inter-brain synchrony (*.pkl files).

                        * The structure of pkl files is each pair will have 4 lists, which has the following order.
                        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,\
                        then move the 2nd four lists which belong to pair 2 and so on.    
                    #. Indices of deleted epochs of EEG (``*.pkl`` files)

                        * Length of list, once pkl file is loaded, is equal to the number of pairs.
                        * For instance, if we have 13 pairs, then there will be 13 lists within that ``*.pkl`` file.
            """
        
        odd_subject_direct_post_suffix = (
            "-direct_post_right_left_point_combined_raw.fif"
        )
        even_subject_direct_post_suffix = (
            "-direct_post_left_right_point_combined_raw.fif"
        )

        start = timer()

        all_deleted_epochs_indices_direct_post = []

        total_n_connections_all_pairs_direct_post = []

        list_scores_all_theta = []
        list_scores_all_alpha = []
        list_scores_all_beta = []
        list_scores_all_gamma = []
        list_scores_all_direct_post = []

        total_n_connections_theta = []
        total_n_connections_alpha = []
        total_n_connections_beta = []
        total_n_connections_gamma = []
        total_n_connections_all_pairs_direct_post = []

        # Note : no_filter = No statistical analysis yet conducted to see which connections that are significant
        list_direct_post_no_filter_all = []

        original_bad_channels_all = self._bad_channels

        # To loop subject number.
        # Every loop there are two files that are taken (odd-even subject)

        begin = 0
        end = self.number_raw_files
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
        ):

            os.chdir(experimental_data)

            # If we want to exclude pair that wants to be skipped
            # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
            # Uncomment this line, in case you have files that want to be skipped
            # if (i == 2):  # NOTE: Indicate pair
            #     continue

            # Subject no. 1 - 10
            if (i + 1) <= 9:
                fname1_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 1)
                    + odd_subject_direct_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 2)
                    + even_subject_direct_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 10:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_direct_post_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 11 - 20
            elif (i + 1) >= 11:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_direct_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_direct_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 20:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_direct_post_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 21 - 30
            elif (i + 1) >= 21:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_direct_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_direct_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 30:
                    fname2_direct = (
                        experimental_data
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

            raw1_direct = mne.io.read_raw_fif(
                fname_S1_direct, preload=True, verbose=False
            )
            raw2_direct = mne.io.read_raw_fif(
                fname_S2_direct, preload=True, verbose=False
            )

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
            (
                cleaned_epochs_AR,
                percentage_rejected_epoch,
                delete_epochs_indices,
            ) = prep.AR_local(cleaned_epochs_ICA, verbose=True)

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
                preproc_S1,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            psd2 = analyses.pow(
                preproc_S2,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            data_psd = np.array([psd1.psd, psd2.psd])

            #
            # # Connectivity
            # with ICA
            data_inter = np.array([preproc_S1, preproc_S2])
            # Computing analytic signal per frequency band
            # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
            complex_signal = analyses.compute_freq_bands(
                data_inter, sampling_rate, freq_bands
            )
            # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
            result = analyses.compute_sync(complex_signal, mode=self.__algorithm)
            #
            # NOTE: Slicing results to get the Inter-brain part of the matrix.
            # Refer to this for slicing or counting no. of connections later on
            #
            n_ch = len(preproc_S1.info["ch_names"])
            theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

            list_direct_post_no_filter_all.append(theta)
            list_direct_post_no_filter_all.append(alpha)
            list_direct_post_no_filter_all.append(beta)
            list_direct_post_no_filter_all.append(gamma)

            # Check if inter-brain connection scores have been put into a list
            print(
                f"(direct_post) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
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
            os.chdir(preprocessed_data)

            # Save pre-processed (epoched) data of subject 1
            preproc_S1.save(epoched_file_name_S1, overwrite=True)

            # Save pre-processed (epoched) data of subject 2
            preproc_S2.save(epoched_file_name_S2, overwrite=True)

        # Change to a directory where we want to save the above populated lists (pre-processed data)
        os.chdir(sync_score_n_del_indices)

        # Save the scores of inter-brain synchrony from each pair into pkl file
        filename_sync = (
            "List_" + self.__algorithm + "_scores_all_pairs_direct_post_no_filter.pkl"
        )

        with open(filename_sync, "wb") as handle:
            pickle.dump(
                list_direct_post_no_filter_all,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # NOTE : The structure of files is each pair will have 4 lists, which has the following order
        #        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
        #        * then move the 2nd four lists which belong to pair 2 and so on.
        print(
            "(direct_post) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
        )

        # Save indices of deleted epochs from each pair into pkl file
        # NOTE : Length of list once pkl file is loaded is equal to the number of pairs
        # If we have 15 pairs, then there will be 15 lists within that pkl file
        with open("list_deleted_epoch_indices_direct_post.pkl", "wb") as handle:
            pickle.dump(
                all_deleted_epochs_indices_direct_post,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(
            "(direct_post) All indices of deleted epochs have been saved into a pickle file"
        )

        # Count elapsed time
        end = timer()
        # Calling function to convert seconds to hour minute, second
        print(f"Processed time : {self.__intermediary_clean_eeg.convert(end - start)}")  # type: ignore

    def clean_N_compute_sync_eeg_natural_pre_exp(
        self,
        experimental_data: str,
        preprocessed_data: str,
        sync_score_n_del_indices: str,
    ):
        """
            #. Clean up noises of EEG files that have been combined
            #. Delete bad epochs
            #. Calculate score of inter-brain synchrony


            :param experimental_data: folder where the combined files (natural_pre) of experiment that will be cleaned are stored(Subject 1 and 2).
            :type experimental_data: str
            :param preprocessed_data: folder to which will store the pre-processed epoched data(*.fif).
            :type preprocessed_data: str
            :param sync_score_n_del_indices: folder to which will store the indices of deleted epoch of EEG data.
            :type sync_score_n_del_indices: str
            :returns:
                #. Pre-processed epoched EEG file of subject 1 (*.fif)
                #. Pre-processed epoched EEG file of subject 2 (*.fif)
                #. Scores of inter-brain synchrony (*.pkl files)
            :rtype: ``*.fif`` (mne) and ``*.pkl``

            .. note::
                returns:
                    #. Pre-processed epoched EEG file of subject 1 (``*.fif``).
                    #. Pre-processed epoched EEG file of subject 2 (``*.fif``).
                    #. Scores of inter-brain synchrony (*.pkl files).

                        * The structure of pkl files is each pair will have 4 lists, which has the following order.
                        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,\
                        then move the 2nd four lists which belong to pair 2 and so on.    
                    #. Indices of deleted epochs of EEG (``*.pkl`` files)

                        * Length of list, once pkl file is loaded, is equal to the number of pairs.
                        * For instance, if we have 13 pairs, then there will be 13 lists within that ``*.pkl`` file.
        """
        
        odd_subject_natural_pre_suffix = (
            "-natural_pre_right_left_point_combined_raw.fif"
        )
        even_subject_natural_pre_suffix = (
            "-natural_pre_left_right_point_combined_raw.fif"
        )

        start = timer()

        all_deleted_epochs_indices_natural_pre = []

        total_n_connections_all_pairs_natural_pre = []

        list_scores_all_theta = []
        list_scores_all_alpha = []
        list_scores_all_beta = []
        list_scores_all_gamma = []
        list_scores_all_natural_pre = []

        total_n_connections_theta = []
        total_n_connections_alpha = []
        total_n_connections_beta = []
        total_n_connections_gamma = []
        total_n_connections_all_pairs_natural_pre = []

        # Note : no_filter = No statistical analysis yet conducted to see which connections that are significant
        list_natural_pre_no_filter_all = []

        original_bad_channels_all = self._bad_channels

        # To loop subject number.
        # Every loop there are two files that are taken (odd-even subject)

        begin = 0
        end = self.number_raw_files
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
        ):

            os.chdir(experimental_data)

            # If we want to exclude pair that wants to be skipped
            # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
            # Uncomment this line, in case you have files that want to be skipped
            # if (i == 2):  # NOTE: Indicate pair
            #     continue

            # Subject no. 1 - 10
            if (i + 1) <= 9:
                fname1_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 1)
                    + odd_subject_natural_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 2)
                    + even_subject_natural_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 10:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_natural_pre_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 11 - 20
            elif (i + 1) >= 11:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_natural_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_natural_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 20:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_natural_pre_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 21 - 30
            elif (i + 1) >= 21:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_natural_pre_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_natural_pre_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 30:
                    fname2_direct = (
                        experimental_data
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

            fname_S1_direct = fname1_direct
            fname_S2_direct = fname2_direct

            # Get original bad channels for odd and even subject
            original_bad_channels1 = original_bad_channels_all[i]
            original_bad_channels2 = original_bad_channels_all[i + 1]

            raw1_direct = mne.io.read_raw_fif(
                fname_S1_direct, preload=True, verbose=False
            )
            raw2_direct = mne.io.read_raw_fif(
                fname_S2_direct, preload=True, verbose=False
            )

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
            (
                cleaned_epochs_AR,
                percentage_rejected_epoch,
                delete_epochs_indices,
            ) = prep.AR_local(cleaned_epochs_ICA, verbose=True)

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
                preproc_S1,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            psd2 = analyses.pow(
                preproc_S2,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            data_psd = np.array([psd1.psd, psd2.psd])

            #
            # # Connectivity
            # with ICA
            data_inter = np.array([preproc_S1, preproc_S2])
            # Computing analytic signal per frequency band
            # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
            complex_signal = analyses.compute_freq_bands(
                data_inter, sampling_rate, freq_bands
            )
            # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
            result = analyses.compute_sync(complex_signal, mode=self.__algorithm)
            #
            # NOTE: Slicing results to get the Inter-brain part of the matrix.
            # Refer to this for slicing or counting no. of connections later on
            #
            n_ch = len(preproc_S1.info["ch_names"])
            theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

            list_natural_pre_no_filter_all.append(theta)
            list_natural_pre_no_filter_all.append(alpha)
            list_natural_pre_no_filter_all.append(beta)
            list_natural_pre_no_filter_all.append(gamma)

            # Check if inter-brain connection scores have been put into a list
            print(
                f"(natural_pre) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
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
            os.chdir(preprocessed_data)

            # Save pre-processed (epoched) data of subject 1
            preproc_S1.save(epoched_file_name_S1, overwrite=True)

            # Save pre-processed (epoched) data of subject 2
            preproc_S2.save(epoched_file_name_S2, overwrite=True)

        # Change to a directory where we want to save the above populated lists (pre-processed data)
        os.chdir(sync_score_n_del_indices)

        # Save the scores of inter-brain synchrony from each pair into pkl file
        filename_sync = (
            "List_" + self.__algorithm + "_scores_all_pairs_natural_pre_no_filter.pkl"
        )

        with open(filename_sync, "wb") as handle:
            pickle.dump(
                list_natural_pre_no_filter_all,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # NOTE : The structure of files is each pair will have 4 lists, which has the following order
        #        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
        #        * then move the 2nd four lists which belong to pair 2 and so on.
        print(
            "(natural_pre) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
        )

        # Save indices of deleted epochs from each pair into pkl file
        # NOTE : Length of list once pkl file is loaded is equal to the number of pairs
        # If we have 15 pairs, then there will be 15 lists within that pkl file
        with open("list_deleted_epoch_indices_natural_pre.pkl", "wb") as handle:
            pickle.dump(
                all_deleted_epochs_indices_natural_pre,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(
            "(natural_pre) All indices of deleted epochs have been saved into a pickle file"
        )

        # Count elapsed time
        end = timer()
        # Calling function to convert seconds to hour minute, second
        print(f"Processed time : {self.__intermediary_clean_eeg.convert(end - start)}")  # type: ignore

    def clean_N_compute_sync_eeg_natural_post_exp(
        self,
        experimental_data: str,
        preprocessed_data: str,
        sync_score_n_del_indices: str,
    ):
        """
        Objective : 1. Clean up noises of EEG files that have been combined
                    2. Delete bad epochs
                    3. Calculate score of inter-brain synchrony

        Parameters:
                    - experimental_data (str): Folder where the combined files (natural_post) of experiment that will be cleaned are stored(Subject 1 and 2)

                    - preprocessed_data (str): Folder to which will store the pre-processed epoched data(*.fif)

                    - sync_score_n_del_indices (str): Folder to which will store the indices of deleted epoch of EEG data

        Output    :
                    1. Pre-processed epoched EEG file of subject 1 (*.fif)
                    2. Pre-processed epoched EEG file of subject 2 (*.fif)
                    3. Scores of inter-brain synchrony (*.pkl files)

                        Note : The structure of pkl files is each pair will have 4 lists, which has the following order
                            * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
                            * then move the 2nd four lists which belong to pair 2 and so on.

                    4. Indices of deleted epochs of EEG (*.pkl files)

                        Note : Length of list once pkl file is loaded is equal to the number of pairs
                            If we have 15 pairs, then there will be 15 lists within that pkl file
        """

        odd_subject_natural_post_suffix = (
            "-natural_post_right_left_point_combined_raw.fif"
        )
        even_subject_natural_post_suffix = (
            "-natural_post_left_right_point_combined_raw.fif"
        )

        start = timer()

        all_deleted_epochs_indices_natural_post = []

        total_n_connections_all_pairs_natural_post = []

        list_scores_all_theta = []
        list_scores_all_alpha = []
        list_scores_all_beta = []
        list_scores_all_gamma = []
        list_scores_all_natural_post = []

        total_n_connections_theta = []
        total_n_connections_alpha = []
        total_n_connections_beta = []
        total_n_connections_gamma = []
        total_n_connections_all_pairs_natural_post = []

        # Note : no_filter = No statistical analysis yet conducted to see which connections that are significant
        list_natural_post_no_filter_all = []

        original_bad_channels_all = self._bad_channels

        # To loop subject number.
        # Every loop there are two files that are taken (odd-even subject)

        begin = 0
        end = self.number_raw_files
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Please, listen 2 music & have some coffee..."
        ):

            os.chdir(experimental_data)

            # If we want to exclude pair that wants to be skipped
            # For example, if pair 2 is bad, then files of subject no.3 & 4 (the data is not good) will not be processed
            # Uncomment this line, in case you have files that want to be skipped
            # if (i == 2):  # NOTE: Indicate pair
            #     continue

            # Subject no. 1 - 10
            if (i + 1) <= 9:
                fname1_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 1)
                    + odd_subject_natural_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S0"
                    + str(i + 2)
                    + even_subject_natural_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 10:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_natural_post_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 11 - 20
            elif (i + 1) >= 11:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_natural_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_natural_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 20:
                    fname2_direct = (
                        experimental_data
                        + "S"
                        + str(i + 2)
                        + even_subject_natural_post_suffix
                    )

                # Indicator of which files are being processed
                print(f"Processing S-{i + 1} & S-{i + 2}")

            # Subject no. 21 - 30
            elif (i + 1) >= 21:
                fname1_direct = (
                    experimental_data
                    + "S"
                    + str(i + 1)
                    + odd_subject_natural_post_suffix
                )
                fname2_direct = (
                    experimental_data
                    + "S"
                    + str(i + 2)
                    + even_subject_natural_post_suffix
                )
                # Replace fname2_direct variable
                if (i + 2) == 30:
                    fname2_direct = (
                        experimental_data
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

            fname_S1_direct = fname1_direct
            fname_S2_direct = fname2_direct

            # Get original bad channels for odd and even subject
            original_bad_channels1 = original_bad_channels_all[i]
            original_bad_channels2 = original_bad_channels_all[i + 1]

            raw1_direct = mne.io.read_raw_fif(
                fname_S1_direct, preload=True, verbose=False
            )
            raw2_direct = mne.io.read_raw_fif(
                fname_S2_direct, preload=True, verbose=False
            )

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
            (
                cleaned_epochs_AR,
                percentage_rejected_epoch,
                delete_epochs_indices,
            ) = prep.AR_local(cleaned_epochs_ICA, verbose=True)

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
                preproc_S1,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            psd2 = analyses.pow(
                preproc_S2,
                fmin=4,
                fmax=40,
                n_fft=1000,
                n_per_seg=1000,
                epochs_average=True,
            )
            data_psd = np.array([psd1.psd, psd2.psd])

            #
            # # Connectivity
            # with ICA
            data_inter = np.array([preproc_S1, preproc_S2])
            # Computing analytic signal per frequency band
            # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
            complex_signal = analyses.compute_freq_bands(
                data_inter, sampling_rate, freq_bands
            )
            # Computing frequency- and time-frequency-domain connectivity, using circular correlation 'ccorr'
            result = analyses.compute_sync(complex_signal, mode=self.__algorithm)
            #
            # NOTE: Slicing results to get the Inter-brain part of the matrix.
            # Refer to this for slicing or counting no. of connections later on
            #
            n_ch = len(preproc_S1.info["ch_names"])
            theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch : 2 * n_ch]

            list_natural_post_no_filter_all.append(theta)
            list_natural_post_no_filter_all.append(alpha)
            list_natural_post_no_filter_all.append(beta)
            list_natural_post_no_filter_all.append(gamma)

            # Check if inter-brain connection scores have been put into a list
            print(
                f"(natural_post) inter-brain connection scores of S-{i + 1} & S-{i + 2} \
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
            os.chdir(preprocessed_data)

            # Save pre-processed (epoched) data of subject 1
            preproc_S1.save(epoched_file_name_S1, overwrite=True)

            # Save pre-processed (epoched) data of subject 2
            preproc_S2.save(epoched_file_name_S2, overwrite=True)

        # Change to a directory where we want to save the above populated lists (pre-processed data)
        os.chdir(sync_score_n_del_indices)

        # Save the scores of inter-brain synchrony from each pair into pkl file
        filename_sync = (
            "List_" + self.__algorithm + "_scores_all_pairs_natural_post_no_filter.pkl"
        )

        with open(filename_sync, "wb") as handle:
            pickle.dump(
                list_natural_post_no_filter_all,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # NOTE : The structure of files is each pair will have 4 lists, which has the following order
        #        * Theta, Alpha, Beta, and Gamma. So for example, the first 4 lists are belonged to pair 1,
        #        * then move the 2nd four lists which belong to pair 2 and so on.
        print(
            "(natural_post) All inter-brain synchrony scores (theta, alpha, beta, gamma) of all pairs have been saved into a pickle file"
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
        print(
            "(natural_post) All indices of deleted epochs have been saved into a pickle file"
        )

        # Count elapsed time
        end = timer()
        # Calling function to convert seconds to hour minute, second
        print(f"Processed time : {self.__intermediary_clean_eeg.convert(end - start)}")  # type: ignore
