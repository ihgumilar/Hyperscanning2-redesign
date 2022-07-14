import statistics
import os
from icecream import ic
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import scipy.stats as stats
import pickle
from hypyp import viz
from hypyp import stats
from hypyp import analyses
from hypyp import prep
import mne
from hypyp.ext.mpl3d.camera import Camera
from hypyp.ext.mpl3d.mesh import Mesh
from hypyp.ext.mpl3d import glm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from LabelConverter import convert_ihshan_paired_channel_label
import pandas as pd
import csv
import scipy
import numpy as np
import requests
from collections import OrderedDict
from copy import copy
import io
from tqdm import tqdm
import warnings
from timeit import default_timer as timer
warnings.filterwarnings("ignore")

# %% Time conversion
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)
# %% markdown
Direct eye(Pre - training)
# %%
# Container for no. of connections of ALL participants
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

# Bad channels
original_bad_channels_all = [['FP1', 'C3', 'T7'], ['FP1', 'F7', 'C4'], ['FP1', 'Fp2', 'F7', 'C4'],
                         ['FP1'],  ['FP1', 'Fp2', 'F7', 'C4', 'P4'], [
                             'FP1', 'Fp2', 'F7', 'C4', 'P4'], [],
                         ['Fp2', 'C3'], ['F3'], ['Fp2', 'F4', 'C3', 'P3'],
                         ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4'],
                         ['FP1', 'T7', 'C3', 'P4'], [], [
                             'Fp2', 'C3', 'P3'], [],
                         ['Fp2', 'F3', 'C3'], ['F7', 'F3', 'T7', 'P8'],
                         ['Fp2', 'C3', 'P3', 'P4', 'O1'], [], [
                             'Fp2', 'C3'], ['P7'],
                         ['Fp2', 'C3', 'O1'], ['P3', 'P4'], [
                             'Fp2', 'C3', 'P4'], [],
                         ['Fp2', 'C3'], ['P3', 'P4'], [
                             'Fp2', 'C3'], [], ['Fp2', 'C3'],
                         ['P3', 'P4'], ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4']]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)
begin = 0
end = 32
step = 2

for i in tqdm(range(begin, end, step), desc="Please, listen 2 music & have some coffee..."):

    # NOTE: Exclude pair 2 - file of subject no.3 & 4 (the data is not good)
    if (i == 2):  # NOTE: Indicate pair
        continue
    fname1_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 1) + "-direct_pre_right_left_point_combined_raw.fif"
    # fname1_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S1-direct_pre_right_left_point_combined_raw.fif"
    fname2_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 2) + "-direct_pre_left_right_point_combined_raw.fif"
    # fname2_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S2-direct_pre_left_right_point_combined_raw.fif"

    freq_bands = {'Theta': [4, 7],
                  'Alpha': [7.5, 13],
                  'Beta': [13.5, 29.5],
                  'Gamma': [30, 40]}

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_direct = fname1_direct
    fname_S2_direct = fname2_direct

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]
    # original_bad_channels1 = original_bad_channels_all[0]
    # original_bad_channels2 = original_bad_channels_all[1]

    # os.getcwd() # delete this
    # os.chdir('/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/')

    raw1_direct = mne.io.read_raw_fif(
        fname_S1_direct, preload=True, verbose=False)
    raw2_direct = mne.io.read_raw_fif(
        fname_S2_direct, preload=True, verbose=False)

    raw1_direct.info['bads'] = original_bad_channels1
    raw2_direct.info['bads'] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_direct = raw1_direct.filter(l_freq=1, h_freq=40)
    raw2_direct = raw2_direct.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_direct = raw1_direct.copy().interpolate_bads(reset_bads=True)
    raw2_direct = raw2_direct.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_direct, id=1, duration=1)

    # Epoch length is 1 second
    epo1_direct = mne.Epochs(raw1_direct, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)
    epo2_direct = mne.Epochs(raw2_direct, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)

    mne.epochs.equalize_epoch_counts([epo1_direct, epo2_direct])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_direct.info['sfreq']
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit([epo1_direct, epo2_direct],
                        n_components=16,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)

    cleaned_epochs_ICA = prep.ICA_choice_comp(
        icas, [epo1_direct, epo2_direct])


    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # TODO: Populate indices of bad epochs into a list
    cleaned_epochs_AR, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=False)

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

    psd1 = analyses.pow(preproc_S1, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    psd2 = analyses.pow(preproc_S2, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # ### Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                 freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, 'plv'
    result = analyses.compute_sync(complex_signal, mode='ccorr')
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info['ch_names'])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch:2 * n_ch]

    list_circular_correlation_direct_pre_no_filter_all.append(theta)
    list_circular_correlation_direct_pre_no_filter_all.append(alpha)
    list_circular_correlation_direct_pre_no_filter_all.append(beta)
    list_circular_correlation_direct_pre_no_filter_all.append(gamma)

with open('list_circular_correlation_scores_all_pairs_direct_pre_no_filter.pkl', 'wb') as handle:
    pickle.dump(list_circular_correlation_direct_pre_no_filter_all, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")
#
# %% Save real power-correlation scores into a list
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
# %% markdown
Direct eye(Post - training)
# %%
# Container for no. of connections of ALL participants
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

# Bad channels
original_bad_channels_all = [['FP1', 'C3', 'T7'], ['FP1', 'F7', 'C4'], ['FP1', 'Fp2', 'F7', 'C4'],
                         ['FP1'],  ['FP1', 'Fp2', 'F7', 'C4', 'P4'], [
                             'FP1', 'Fp2', 'F7', 'C4', 'P4'], [],
                         ['Fp2', 'C3'], ['F3'], ['Fp2', 'F4', 'C3', 'P3'],
                         ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4'],
                         ['FP1', 'T7', 'C3', 'P4'], [], [
                             'Fp2', 'C3', 'P3'], [],
                         ['Fp2', 'F3', 'C3'], ['F7', 'F3', 'T7', 'P8'],
                         ['Fp2', 'C3', 'P3', 'P4', 'O1'], [], [
                             'Fp2', 'C3'], ['P7'],
                         ['Fp2', 'C3', 'O1'], ['P3', 'P4'], [
                             'Fp2', 'C3', 'P4'], [],
                         ['Fp2', 'C3'], ['P3', 'P4'], [
                             'Fp2', 'C3'], [], ['Fp2', 'C3'],
                         ['P3', 'P4'], ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4']]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)
begin = 0
end = 32
step = 2

for i in tqdm(range(begin, end, step), desc="Please, listen 2 music & have some coffee..."):

    # NOTE: Exclude pair 2 - file of subject no.3 & 4 (the data is not good)
    if (i == 2):  # NOTE: Indicate pair
        continue
    fname1_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 1) + "-direct_post_right_left_point_combined_raw.fif"
    # fname1_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S1-direct_post_right_left_point_combined_raw.fif"
    fname2_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 2) + "-direct_post_left_right_point_combined_raw.fif"
    # fname2_direct = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S2-direct_post_left_right_point_combined_raw.fif"

    freq_bands = {'Theta': [4, 7],
                  'Alpha': [7.5, 13],
                  'Beta': [13.5, 29.5],
                  'Gamma': [30, 40]}

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_direct = fname1_direct
    fname_S2_direct = fname2_direct

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]
    # original_bad_channels1 = original_bad_channels_all[0]
    # original_bad_channels2 = original_bad_channels_all[1]

    # os.getcwd() # delete this
    # os.chdir('/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/')

    raw1_direct = mne.io.read_raw_fif(
        fname_S1_direct, preload=True, verbose=False)
    raw2_direct = mne.io.read_raw_fif(
        fname_S2_direct, preload=True, verbose=False)

    raw1_direct.info['bads'] = original_bad_channels1
    raw2_direct.info['bads'] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_direct = raw1_direct.filter(l_freq=1, h_freq=40)
    raw2_direct = raw2_direct.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_direct = raw1_direct.copy().interpolate_bads(reset_bads=True)
    raw2_direct = raw2_direct.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_direct, id=1, duration=1)

    # Epoch length is 1 second
    epo1_direct = mne.Epochs(raw1_direct, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)
    epo2_direct = mne.Epochs(raw2_direct, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)

    mne.epochs.equalize_epoch_counts([epo1_direct, epo2_direct])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_direct.info['sfreq']
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit([epo1_direct, epo2_direct],
                        n_components=16,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)

    cleaned_epochs_ICA = prep.ICA_choice_comp(
        icas, [epo1_direct, epo2_direct])


    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # TODO: Populate indices of bad epochs into a list
    cleaned_epochs_AR, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=False)

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

    psd1 = analyses.pow(preproc_S1, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    psd2 = analyses.pow(preproc_S2, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # ### Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                 freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, 'plv'
    result = analyses.compute_sync(complex_signal, mode='ccorr')
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info['ch_names'])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch:2 * n_ch]

    list_circular_correlation_direct_post_no_filter_all.append(theta)
    list_circular_correlation_direct_post_no_filter_all.append(alpha)
    list_circular_correlation_direct_post_no_filter_all.append(beta)
    list_circular_correlation_direct_post_no_filter_all.append(gamma)

with open('list_circular_correlation_scores_all_pairs_direct_post_no_filter.pkl', 'wb') as handle:
    pickle.dump(list_circular_correlation_direct_post_no_filter_all, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# a = result[0:n_ch, n_ch:2 * n_ch]
# %% markdown
Averted eye(Pre - training)

# %%
# Container for no. of connections of ALL participants
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

# Bad channels
original_bad_channels_all = [['FP1', 'C3', 'T7'], ['FP1', 'F7', 'C4'], ['FP1', 'Fp2', 'F7', 'C4'],
                         ['FP1'],  ['FP1', 'Fp2', 'F7', 'C4', 'P4'], [
                             'FP1', 'Fp2', 'F7', 'C4', 'P4'], [],
                         ['Fp2', 'C3'], ['F3'], ['Fp2', 'F4', 'C3', 'P3'],
                         ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4'],
                         ['FP1', 'T7', 'C3', 'P4'], [], [
                             'Fp2', 'C3', 'P3'], [],
                         ['Fp2', 'F3', 'C3'], ['F7', 'F3', 'T7', 'P8'],
                         ['Fp2', 'C3', 'P3', 'P4', 'O1'], [], [
                             'Fp2', 'C3'], ['P7'],
                         ['Fp2', 'C3', 'O1'], ['P3', 'P4'], [
                             'Fp2', 'C3', 'P4'], [],
                         ['Fp2', 'C3'], ['P3', 'P4'], [
                             'Fp2', 'C3'], [], ['Fp2', 'C3'],
                         ['P3', 'P4'], ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4']]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)
begin = 0
end = 32
step = 2

for i in tqdm(range(begin, end, step), desc="Please, listen 2 music & have some coffee..."):

    # NOTE: Exclude pair 2 - file of subject no.3 & 4 (the data is not good)
    if (i == 2):  # NOTE: Indicate pair
        continue
    fname1_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 1) + "-averted_pre_right_left_point_combined_raw.fif"
    # fname1_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S1-averted_pre_right_left_point_combined_raw.fif"
    fname2_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 2) + "-averted_pre_left_right_point_combined_raw.fif"
    # fname2_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S2-averted_pre_left_right_point_combined_raw.fif"

    freq_bands = {'Theta': [4, 7],
                  'Alpha': [7.5, 13],
                  'Beta': [13.5, 29.5],
                  'Gamma': [30, 40]}

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_averted = fname1_averted
    fname_S2_averted = fname2_averted

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]
    # original_bad_channels1 = original_bad_channels_all[0]
    # original_bad_channels2 = original_bad_channels_all[1]

    # os.getcwd() # delete this
    # os.chdir('/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/')

    raw1_averted = mne.io.read_raw_fif(
        fname_S1_averted, preload=True, verbose=False)
    raw2_averted = mne.io.read_raw_fif(
        fname_S2_averted, preload=True, verbose=False)

    raw1_averted.info['bads'] = original_bad_channels1
    raw2_averted.info['bads'] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_averted = raw1_averted.filter(l_freq=1, h_freq=40)
    raw2_averted = raw2_averted.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_averted = raw1_averted.copy().interpolate_bads(reset_bads=True)
    raw2_averted = raw2_averted.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_averted, id=1, duration=1)

    # Epoch length is 1 second
    epo1_averted = mne.Epochs(raw1_averted, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)
    epo2_averted = mne.Epochs(raw2_averted, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)

    mne.epochs.equalize_epoch_counts([epo1_averted, epo2_averted])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_averted.info['sfreq']
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit([epo1_averted, epo2_averted],
                        n_components=16,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)

    cleaned_epochs_ICA = prep.ICA_choice_comp(
        icas, [epo1_averted, epo2_averted])


    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # TODO: Populate indices of bad epochs into a list
    cleaned_epochs_AR, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=False)

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

    psd1 = analyses.pow(preproc_S1, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    psd2 = analyses.pow(preproc_S2, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # ### Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                 freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, 'plv'
    result = analyses.compute_sync(complex_signal, mode='ccorr')
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info['ch_names'])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch:2 * n_ch]

    list_circular_correlation_averted_pre_no_filter_all.append(theta)
    list_circular_correlation_averted_pre_no_filter_all.append(alpha)
    list_circular_correlation_averted_pre_no_filter_all.append(beta)
    list_circular_correlation_averted_pre_no_filter_all.append(gamma)

with open('list_circular_correlation_scores_all_pairs_averted_pre_no_filter.pkl', 'wb') as handle:
    pickle.dump(list_circular_correlation_averted_pre_no_filter_all, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")
#
# %% markdown
Averted eye(Post - training)
# %%
# Container for no. of connections of ALL participants
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

# Bad channels
original_bad_channels_all = [['FP1', 'C3', 'T7'], ['FP1', 'F7', 'C4'], ['FP1', 'Fp2', 'F7', 'C4'],
                         ['FP1'],  ['FP1', 'Fp2', 'F7', 'C4', 'P4'], [
                             'FP1', 'Fp2', 'F7', 'C4', 'P4'], [],
                         ['Fp2', 'C3'], ['F3'], ['Fp2', 'F4', 'C3', 'P3'],
                         ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4'],
                         ['FP1', 'T7', 'C3', 'P4'], [], [
                             'Fp2', 'C3', 'P3'], [],
                         ['Fp2', 'F3', 'C3'], ['F7', 'F3', 'T7', 'P8'],
                         ['Fp2', 'C3', 'P3', 'P4', 'O1'], [], [
                             'Fp2', 'C3'], ['P7'],
                         ['Fp2', 'C3', 'O1'], ['P3', 'P4'], [
                             'Fp2', 'C3', 'P4'], [],
                         ['Fp2', 'C3'], ['P3', 'P4'], [
                             'Fp2', 'C3'], [], ['Fp2', 'C3'],
                         ['P3', 'P4'], ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4']]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)
begin = 0
end = 32
step = 2

for i in tqdm(range(begin, end, step), desc="Please, listen 2 music & have some coffee..."):

    # NOTE: Exclude pair 2 - file of subject no.3 & 4 (the data is not good)
    if (i == 2):  # NOTE: Indicate pair
        continue
    fname1_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 1) + "-averted_post_right_left_point_combined_raw.fif"
    # fname1_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S1-averted_post_right_left_point_combined_raw.fif"
    fname2_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 2) + "-averted_post_left_right_point_combined_raw.fif"
    # fname2_averted = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S2-averted_post_left_right_point_combined_raw.fif"

    freq_bands = {'Theta': [4, 7],
                  'Alpha': [7.5, 13],
                  'Beta': [13.5, 29.5],
                  'Gamma': [30, 40]}

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_averted = fname1_averted
    fname_S2_averted = fname2_averted

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]
    # original_bad_channels1 = original_bad_channels_all[0]
    # original_bad_channels2 = original_bad_channels_all[1]

    # os.getcwd() # delete this
    # os.chdir('/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/')

    raw1_averted = mne.io.read_raw_fif(
        fname_S1_averted, preload=True, verbose=False)
    raw2_averted = mne.io.read_raw_fif(
        fname_S2_averted, preload=True, verbose=False)

    raw1_averted.info['bads'] = original_bad_channels1
    raw2_averted.info['bads'] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_averted = raw1_averted.filter(l_freq=1, h_freq=40)
    raw2_averted = raw2_averted.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_averted = raw1_averted.copy().interpolate_bads(reset_bads=True)
    raw2_averted = raw2_averted.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_averted, id=1, duration=1)

    # Epoch length is 1 second
    epo1_averted = mne.Epochs(raw1_averted, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)
    epo2_averted = mne.Epochs(raw2_averted, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)

    mne.epochs.equalize_epoch_counts([epo1_averted, epo2_averted])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_averted.info['sfreq']
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit([epo1_averted, epo2_averted],
                        n_components=16,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)

    cleaned_epochs_ICA = prep.ICA_choice_comp(
        icas, [epo1_averted, epo2_averted])


    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # TODO: Populate indices of bad epochs into a list
    cleaned_epochs_AR, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=False)

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

    psd1 = analyses.pow(preproc_S1, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    psd2 = analyses.pow(preproc_S2, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # ### Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                 freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, 'plv'
    result = analyses.compute_sync(complex_signal, mode='ccorr')
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info['ch_names'])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch:2 * n_ch]

    list_circular_correlation_averted_post_no_filter_all.append(theta)
    list_circular_correlation_averted_post_no_filter_all.append(alpha)
    list_circular_correlation_averted_post_no_filter_all.append(beta)
    list_circular_correlation_averted_post_no_filter_all.append(gamma)

with open('list_circular_correlation_scores_all_pairs_averted_post_no_filter.pkl', 'wb') as handle:
    pickle.dump(list_circular_correlation_averted_post_no_filter_all, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% markdown
Natural eye(Pre - training)

# %%
# Container for no. of connections of ALL participants
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

# Bad channels
original_bad_channels_all = [['FP1', 'C3', 'T7'], ['FP1', 'F7', 'C4'], ['FP1', 'Fp2', 'F7', 'C4'],
                         ['FP1'],  ['FP1', 'Fp2', 'F7', 'C4', 'P4'], [
                             'FP1', 'Fp2', 'F7', 'C4', 'P4'], [],
                         ['Fp2', 'C3'], ['F3'], ['Fp2', 'F4', 'C3', 'P3'],
                         ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4'],
                         ['FP1', 'T7', 'C3', 'P4'], [], [
                             'Fp2', 'C3', 'P3'], [],
                         ['Fp2', 'F3', 'C3'], ['F7', 'F3', 'T7', 'P8'],
                         ['Fp2', 'C3', 'P3', 'P4', 'O1'], [], [
                             'Fp2', 'C3'], ['P7'],
                         ['Fp2', 'C3', 'O1'], ['P3', 'P4'], [
                             'Fp2', 'C3', 'P4'], [],
                         ['Fp2', 'C3'], ['P3', 'P4'], [
                             'Fp2', 'C3'], [], ['Fp2', 'C3'],
                         ['P3', 'P4'], ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4']]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)
begin = 0
end = 32
step = 2

for i in tqdm(range(begin, end, step), desc="Please, listen 2 music & have some coffee..."):

    # NOTE: Exclude pair 2 - file of subject no.3 & 4 (the data is not good)
    if (i == 2):  # NOTE: Indicate pair
        continue
    fname1_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 1) + "-natural_pre_right_left_point_combined_raw.fif"
    # fname1_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S1-natural_pre_right_left_point_combined_raw.fif"
    fname2_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 2) + "-natural_pre_left_right_point_combined_raw.fif"
    # fname2_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S2-natural_pre_left_right_point_combined_raw.fif"

    freq_bands = {'Theta': [4, 7],
                  'Alpha': [7.5, 13],
                  'Beta': [13.5, 29.5],
                  'Gamma': [30, 40]}

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_natural = fname1_natural
    fname_S2_natural = fname2_natural

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]
    # original_bad_channels1 = original_bad_channels_all[0]
    # original_bad_channels2 = original_bad_channels_all[1]

    # os.getcwd() # delete this
    # os.chdir('/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/')

    raw1_natural = mne.io.read_raw_fif(
        fname_S1_natural, preload=True, verbose=False)
    raw2_natural = mne.io.read_raw_fif(
        fname_S2_natural, preload=True, verbose=False)

    raw1_natural.info['bads'] = original_bad_channels1
    raw2_natural.info['bads'] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_natural = raw1_natural.filter(l_freq=1, h_freq=40)
    raw2_natural = raw2_natural.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_natural = raw1_natural.copy().interpolate_bads(reset_bads=True)
    raw2_natural = raw2_natural.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_natural, id=1, duration=1)

    # Epoch length is 1 second
    epo1_natural = mne.Epochs(raw1_natural, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)
    epo2_natural = mne.Epochs(raw2_natural, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)

    mne.epochs.equalize_epoch_counts([epo1_natural, epo2_natural])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_natural.info['sfreq']
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit([epo1_natural, epo2_natural],
                        n_components=16,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)

    cleaned_epochs_ICA = prep.ICA_choice_comp(
        icas, [epo1_natural, epo2_natural])


    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # TODO: Populate indices of bad epochs into a list
    cleaned_epochs_AR, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=False)

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

    psd1 = analyses.pow(preproc_S1, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    psd2 = analyses.pow(preproc_S2, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # ### Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                 freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, 'plv'
    result = analyses.compute_sync(complex_signal, mode='ccorr')
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info['ch_names'])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch:2 * n_ch]

    list_circular_correlation_natural_pre_no_filter_all.append(theta)
    list_circular_correlation_natural_pre_no_filter_all.append(alpha)
    list_circular_correlation_natural_pre_no_filter_all.append(beta)
    list_circular_correlation_natural_pre_no_filter_all.append(gamma)

with open('list_circular_correlation_scores_all_pairs_natural_pre_no_filter.pkl', 'wb') as handle:
    pickle.dump(list_circular_correlation_natural_pre_no_filter_all, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")

# %% markdown
Natural eye(Post - training)
# %%
# Container for no. of connections of ALL participants
start = timer()

all_deleted_epochs_indices_natural_post = []

total_n_connections_all_pairs_natural_post = []

list_real_circular_correlation_all_theta = []
list_real_circular_correlation_all_alpha = []
list_real_circular_correlation_all_beta = []
list_real_circular_correlation_all_gamma = []
list_real_circular_correlation_all_natural_post = []

total_n_connections_theta = []
total_n_connections_alpha = []
total_n_connections_beta = []
total_n_connections_gamma = []
total_n_connections_all_pairs_natural_post = []

list_real_plv_natural_post_no_filter_all = []

# Bad channels
original_bad_channels_all = [['FP1', 'C3', 'T7'], ['FP1', 'F7', 'C4'], ['FP1', 'Fp2', 'F7', 'C4'],
                         ['FP1'],  ['FP1', 'Fp2', 'F7', 'C4', 'P4'], [
                             'FP1', 'Fp2', 'F7', 'C4', 'P4'], [],
                         ['Fp2', 'C3'], ['F3'], ['Fp2', 'F4', 'C3', 'P3'],
                         ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4'],
                         ['FP1', 'T7', 'C3', 'P4'], [], [
                             'Fp2', 'C3', 'P3'], [],
                         ['Fp2', 'F3', 'C3'], ['F7', 'F3', 'T7', 'P8'],
                         ['Fp2', 'C3', 'P3', 'P4', 'O1'], [], [
                             'Fp2', 'C3'], ['P7'],
                         ['Fp2', 'C3', 'O1'], ['P3', 'P4'], [
                             'Fp2', 'C3', 'P4'], [],
                         ['Fp2', 'C3'], ['P3', 'P4'], [
                             'Fp2', 'C3'], [], ['Fp2', 'C3'],
                         ['P3', 'P4'], ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4']]


# To loop subject number.
# Every loop there are two files that are taken (odd-even subject)
begin = 0
end = 32
step = 2

for i in tqdm(range(begin, end, step), desc="Please, listen 2 music & have some coffee..."):

    # NOTE: Exclude pair 2 - file of subject no.3 & 4 (the data is not good)
    if (i == 2):  # NOTE: Indicate pair
        continue
    fname1_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 1) + "-natural_post_right_left_point_combined_raw.fif"
    # fname1_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S1-natural_post_right_left_point_combined_raw.fif"
    fname2_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S" + \
        str(i + 2) + "-natural_post_left_right_point_combined_raw.fif"
    # fname2_natural = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/S2-natural_post_left_right_point_combined_raw.fif"

    freq_bands = {'Theta': [4, 7],
                  'Alpha': [7.5, 13],
                  'Beta': [13.5, 29.5],
                  'Gamma': [30, 40]}

    freq_bands = OrderedDict(freq_bands)

    # Container for no. of connections of each band from ONE participant
    # total_n_connections_theta = []

    fname_S1_natural = fname1_natural
    fname_S2_natural = fname2_natural

    # Get original bad channels for odd and even subject
    original_bad_channels1 = original_bad_channels_all[i]
    original_bad_channels2 = original_bad_channels_all[i + 1]
    # original_bad_channels1 = original_bad_channels_all[0]
    # original_bad_channels2 = original_bad_channels_all[1]

    # os.getcwd() # delete this
    # os.chdir('/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/')

    raw1_natural = mne.io.read_raw_fif(
        fname_S1_natural, preload=True, verbose=False)
    raw2_natural = mne.io.read_raw_fif(
        fname_S2_natural, preload=True, verbose=False)

    raw1_natural.info['bads'] = original_bad_channels1
    raw2_natural.info['bads'] = original_bad_channels2

    # Filter raw data (The signal that is retained between 1 - 40 Hz)
    raw1_natural = raw1_natural.filter(l_freq=1, h_freq=40)
    raw2_natural = raw2_natural.filter(l_freq=1, h_freq=40)

    # Interpolate bad channels
    raw1_natural = raw1_natural.copy().interpolate_bads(reset_bads=True)
    raw2_natural = raw2_natural.copy().interpolate_bads(reset_bads=True)

    # Make epochs
    events = mne.make_fixed_length_events(raw1_natural, id=1, duration=1)

    # Epoch length is 1 second
    epo1_natural = mne.Epochs(raw1_natural, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)
    epo2_natural = mne.Epochs(raw2_natural, events, tmin=0.,
                              tmax=1.0, baseline=None, preload=True, verbose=False)

    mne.epochs.equalize_epoch_counts([epo1_natural, epo2_natural])

    # Specify sampling frequency
    # Define here to be used later for computing complex signal

    sampling_rate = epo1_natural.info['sfreq']
    # Preprocessing epochs
    # Computing global AutoReject and Independant Components Analysis for each participant
    icas = prep.ICA_fit([epo1_natural, epo2_natural],
                        n_components=16,
                        method='infomax',
                        fit_params=dict(extended=True),
                        random_state=42)

    cleaned_epochs_ICA = prep.ICA_choice_comp(
        icas, [epo1_natural, epo2_natural])


    # Autoreject
    # Applying local AutoReject for each participant rejecting bad epochs, rejecting or interpolating partially bad channels removing the same bad channels and epochs across participants plotting signal before and after (verbose=True)
    # Auto-reject with ICA

    # TODO: Populate indices of bad epochs into a list
    cleaned_epochs_AR, delete_epochs_indices = prep.AR_local(
        cleaned_epochs_ICA, verbose=False)

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

    psd1 = analyses.pow(preproc_S1, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    psd2 = analyses.pow(preproc_S2, fmin=4, fmax=40,
                        n_fft=1000, n_per_seg=1000, epochs_average=True)
    data_psd = np.array([psd1.psd, psd2.psd])

    #
    # ### Connectivity
    # with ICA
    data_inter = np.array([preproc_S1, preproc_S2])
    # Computing analytic signal per frequency band
    # With ICA (Compute complex signal, that will be used as input for calculationg connectivity, eg. power-correlation score)
    complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                 freq_bands)
    # Computing frequency- and time-frequency-domain connectivity, 'plv'
    result = analyses.compute_sync(complex_signal, mode='ccorr')
    #
    # NOTE: Slicing results to get the Inter-brain part of the matrix.
    # Refer to this for slicing or counting no. of connections later on
    #
    n_ch = len(preproc_S1.info['ch_names'])
    theta, alpha, beta, gamma = result[:, 0:n_ch, n_ch:2 * n_ch]

    list_real_plv_natural_post_no_filter_all.append(theta)
    list_real_plv_natural_post_no_filter_all.append(alpha)
    list_real_plv_natural_post_no_filter_all.append(beta)
    list_real_plv_natural_post_no_filter_all.append(gamma)

with open('list_real_circular_correlation_all_pairs_natural_post_no_filter.pkl', 'wb') as handle:
    pickle.dump(list_real_plv_natural_post_no_filter_all, handle,
                protocol=pickle.HIGHEST_PROTOCOL)

# Count elapsed time
end = timer()
# Calling function to convert seconds to hour minute, second
print(f"Processed time : {convert(end - start)}")
