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
import os
import re
import warnings

import mne
import pandas as pd
from EEG_interfaces import pre_eeg_exp2_redesign
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)


# %% [markdown]
# ## Class of preprocessing exp2-redesign consists of
# 1. extract baseline eeg data
# 1. extract experimental eeg data

# %%
class preproc_exp2_redesign(pre_eeg_exp2_redesign):
    def extract_baseline_eeg_data(
        self,
        path_2_csv_files: str,
        path_2_save_baseline_file: str,
        labelsequence=1,
        bad_files=[],
    ):
        """ * Extract baseline data from raw EEG data which is in csv format. \
            Every raw csv file of EEG data must contain 48 markers in total (opening & closing). \
            Basically, there are 24 markers. However, baseline data is in the first 6 markers. \
            (12 markers if including opening & closing markers).

        :param path_2_csv_files: path to raw EEG file (csv format).
        :type path_2_csv_files: str
        :param path_2_save_baseline_file: path to save extracted baseline data of EEG (output file in *.fif).
        :type path_2_save_baseline_file: str
        :param labelsequence: order of label sequence, in this case is 1 by default, defaults to 1
        :type labelsequence: int, optional
        :param bad_files: raw EEG file(s) that want to be skipped to be processed by the script, defaults to []
        :type bad_files: list, optional
        :returns: extracted_EEG_files (baseline)
        :rtype: *.fif (mne)

        .. note:: * returns: :literal:`*.fif` files  
                       * File name format :
                       * EEG-Subject no_EyeCondition_TrainingCondition_HandCondition_raw.fif.
                           * **EEG-S01-averted_left_tracking_raw**.
                       * There are 6 files in total for each participant.

        .. warning:: All resulted files will be in AVERTED condition \
                     since the baseline condition is in AVERTED condition.
        """

        list_file_names = []
        full_path_2_each_file = []

        for file in os.listdir(path_2_csv_files):

            if file.endswith(".csv"):

                if file in bad_files:

                    # Skip the bad file to be processed
                    print(f"Skipped bad file : {file}")
                    continue

                # Populate all file names only
                list_file_names.append(file)
                list_file_names.sort()

            for i in tqdm(range(len(list_file_names)), desc="In progress"):

                try:
                    labelsequence = int(labelsequence)

                except IOError as err_filename:
                    print(
                        "The format of file name is not correct or file doesn't exist \nThe format must be 'EEG-Sxx.csv' , xx=subject number "
                    )
                    raise
                except ValueError as err_integer:
                    print("The labelsequence input must be integer : ", err_integer)
                    raise

                else:
                    if labelsequence < 1 or labelsequence > 12:
                        print(
                            "The value for labelsequence parameter is out of range. It must be be between 1 and 12"
                        )
                        raise IndexError
                    else:

                        # Load the data
                        fileName = list_file_names[i]
                        print("Processing file : " + list_file_names[i])

                        # Change working directory to which raw EEG files are stored
                        os.chdir(path_2_csv_files)

                        # Read each file by using pandas
                        df = pd.read_csv(fileName, delimiter=",")
                        # Define columns for raw csv file
                        df.columns = [
                            "Index",
                            "FP1",
                            "FP2",
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
                            "X1",
                            "X2",
                            "X3",
                            "X4",
                            "X5",
                            "X6",
                            "X7",
                            "X8",
                            "X9",
                            "X10",
                            "X11",
                            "X12",
                            "X13",
                            "X14",
                            "Marker",
                        ]

                        # Converting all markers (pandas data frame to list)
                        markers = df["Marker"].tolist()

                        indicesOfMarkers = (
                            []
                        )  # Empty list to contain indices of markers
                        for i, c in enumerate(markers):
                            if "9999999" in str(c):
                                indicesOfMarkers.append(i)
                        try:
                            number_markers = len(indicesOfMarkers)
                            if (
                                number_markers != 48
                            ):  # check if the number of markers = 48
                                raise ValueError(
                                    "The {} file has incorrect number of markers : {} ! It MUST be 48".format(
                                        fileName, number_markers
                                    )
                                )
                        except ValueError as err_unmatch_markers:
                            print(err_unmatch_markers)
                            raise

                        # Baseline data used only averted eye condition. Since everything is turned to be white.
                        # Participants did not see each other basically. So, different eye gaze direction does not matter.

                        # This is the order for participants with odd number, e.g., 1, 3, 5, etc..
                        oddOrder1 = [
                            "averted_pre_right_point",
                            "averted_pre_left_point",
                            "averted_left_tracking",
                            "averted_right_tracking",
                            "averted_post_right_point",
                            "averted_post_left_point",
                        ]

                        # This is the order for participants with even number, e.g., 2, 4, 6, etc..
                        evenOrder1 = [
                            "averted_pre_left_point",
                            "averted_pre_right_point",
                            "averted_right_tracking",
                            "averted_left_tracking",
                            "averted_post_left_point",
                            "averted_post_right_point",
                        ]

                        listOfOrders = []
                        listOfOrders.append(oddOrder1)
                        listOfOrders.append(evenOrder1)

                        # It is used to take the above label (oddOrder1 or oddOrder2)
                        i_label_taker = 0
                        if i % 2 == 0:
                            # Even number
                            i_label_taker = 0
                        else:
                            # Odd number
                            i_label_taker = 1

                        chosenOrder = listOfOrders[i_label_taker]

                        ####### BASELINE DATA #######

                        # Get the first 12 markers' indices and extract the data
                        indicesofBaselineMarkers = indicesOfMarkers[:13]

                        # Get the first 12 markers and chunk dataframe based on those indices, and convert it into numpy array
                        # For some data, it can be 13 markers after being extracted because when we combined the data the markers of beginning are
                        # right after the closing marker

                        # Chunk the data based on opening and closing markers and get only the 16 channels data (columns)
                        chunkedData = []
                        for i in range(0, 12, 2):

                            # Convert the data into numpy array type and microvolt (for some reason, the output of OpenBCI is not in microvolt)
                            chunkedData.append(
                                df.iloc[
                                    indicesofBaselineMarkers[
                                        i
                                    ] : indicesofBaselineMarkers[i + 1],
                                    1:17,
                                ].to_numpy()
                                * 1e-6
                            )

                        # Create 16 channels montage 10-20 international standard
                        montage = mne.channels.make_standard_montage("standard_1020")

                        # Pick only 16 channels that are used in Cyton+Daisy OpenBCI
                        # Create info (which is used in MNE-Python)
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
                        info = mne.create_info(
                            ch_names=ch_names, sfreq=125, ch_types=ch_types
                        )
                        info.set_montage("standard_1020", match_case=False)

                        # Match pattern EEG-Sxx (xx = any number)
                        regex = r"\D{3}-S\d+"

                        # Create filenames for *.fif based on the sequence of labels above
                        # filenames_fif = []
                        extracted_file_name_4_baseline = []
                        for i in chosenOrder:
                            # filenames_fif.append(fileName[4:fileName.index(".")] + "-" + i + "_raw.fif")
                            # Find characters (file name) that matches the regex
                            extracted_file_name = re.search(regex, fileName)
                            extracted_file_name_4_baseline.append(
                                fileName[
                                    extracted_file_name.start() : extracted_file_name.end()
                                ]
                                + "-"
                                + i
                                + "_raw.fif"
                            )

                        # Save into *.fif files
                        for i, val in tqdm(
                            enumerate(chunkedData), desc="Saving process..."
                        ):
                            # Load data into MNE-Python
                            baseline_data_needs_label = mne.io.RawArray(
                                val.transpose(), info, verbose=False
                            )
                            # Define a folder where we want to save the baseline data
                            os.chdir(path_2_save_baseline_file)
                            # Save the data in MNE format
                            baseline_data_needs_label.save(
                                extracted_file_name_4_baseline[i], overwrite=True
                            )

            print(
                f"All baseline files have been saved in fif format in this path {path_2_save_baseline_file}"
            )

    def extract_experimental_eeg_data(
        self,
        path_2_csv_files: str,
        path_2_save_experimental_file: str,
        labelsequence_experiment: list,
        bad_files=[],
    ):

        """ * Extract experimental data from raw EEG data which is in csv format.\
            Every raw csv file of EEG data must contain 48 markers in total (opening & closing).\
            Basically, there are 24 markers. However, experimental data is from marker 7 to 24.\
            (36 markers if including opening & closing markers).

        :param path_2_csv_files: path to raw EEG file (csv format).
        :type path_2_csv_files: str
        :param path_2_save_experimental_file: path to save extracted experimental data of EEG (which is in *.fif).
        :type path_2_save_experimental_file: str
        :param labelsequence_experiment: order of label sequence.
        :type labelsequence_experiment: list
        :param bad_files: raw EEG file(s) that want to be skipped to be processed by the script, defaults to []
        :type bad_files: list, optional
        :raises IndexError: _description_
        :raises ValueError: _description_

        .. note:: * returns: :literal:`*.fif` files  
                       * File name format :
                       * EEG-Subject no_EyeCondition__TrainingCondition_HandCondition_raw.fif
                           * **EEG-S01-averted_left_tracking_raw**.
                       * There are 18 files in total for each participant.
        
        """

        list_file_names = []
        full_path_2_each_file = []

        for file in os.listdir(path_2_csv_files):

            if file.endswith(".csv"):

                if file in bad_files:

                    # Skip the bad file to be processed
                    print(f"Skipped bad file : {file}")
                    continue

                # Populate all file names only
                list_file_names.append(file)
                list_file_names.sort()

        for i in tqdm(range(len(list_file_names)), desc="In progress"):

            try:
                labelsequence = int(labelsequence_experiment[i])

            except IOError as err_filename:
                print(
                    "The format of file name is not correct or file doesn't exist \nThe format must be 'EEG-Sxx.csv' , xx=subject number "
                )
                raise
            except ValueError as err_integer:
                print("The labelsequence input must be integer : ", err_integer)
                raise

            else:
                if labelsequence < 1 or labelsequence > 12:
                    print(
                        "The value for labelsequence parameter is out of range. It must be be between 1 and 12"
                    )
                    raise IndexError
                else:

                    # Load the data
                    fileName = list_file_names[i]
                    print("Processing file : " + list_file_names[i])

                    # Change working directory to which raw EEG files are stored
                    os.chdir(path_2_csv_files)

                    # Read each file by using pandas
                    df = pd.read_csv(fileName, delimiter=",")
                    #  Define columns for raw csv file
                    df.columns = [
                        "Index",
                        "FP1",
                        "FP2",
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
                        "X1",
                        "X2",
                        "X3",
                        "X4",
                        "X5",
                        "X6",
                        "X7",
                        "X8",
                        "X9",
                        "X10",
                        "X11",
                        "X12",
                        "X13",
                        "X14",
                        "Marker",
                    ]

                    # Converting all markers (pandas data frame to list)
                    markers = df["Marker"].tolist()

                    indicesOfMarkers = []  # Empty list to contain indices of markers
                    for i, c in enumerate(markers):
                        if "9999999" in str(c):
                            indicesOfMarkers.append(i)
                    try:
                        number_markers = len(indicesOfMarkers)
                        if number_markers != 48:  # check if the number of markers = 48
                            raise ValueError(
                                "The {} file has incorrect number of markers : {} ! It MUST be 48".format(
                                    fileName, number_markers
                                )
                            )
                    except ValueError as err_unmatch_markers:
                        print(err_unmatch_markers)
                        raise

                    # Create a list of labels for experimental data.

                    # Order = 1 (Averted/Direct/Natural)
                    oddOrder1 = [
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                        "direct_pre_right_point",
                        "direct_pre_left_point",
                        "direct_left_tracking",
                        "direct_right_tracking",
                        "direct_post_right_point",
                        "direct_post_left_point",
                        "natural_pre_right_point",
                        "natural_pre_left_point",
                        "natural_left_tracking",
                        "natural_right_tracking",
                        "natural_post_right_point",
                        "natural_post_left_point",
                    ]

                    evenOrder1 = [
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                        "direct_pre_left_point",
                        "direct_pre_right_point",
                        "direct_right_tracking",
                        "direct_left_tracking",
                        "direct_post_left_point",
                        "direct_post_right_point",
                        "natural_pre_left_point",
                        "natural_pre_right_point",
                        "natural_right_tracking",
                        "natural_left_tracking",
                        "natural_post_left_point",
                        "natural_post_right_point",
                    ]

                    # Order = 2 (Averted/Natural/Direct)
                    oddOrder2 = [
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                        "natural_pre_right_point",
                        "natural_pre_left_point",
                        "natural_left_tracking",
                        "natural_right_tracking",
                        "natural_post_right_point",
                        "natural_post_left_point",
                        "direct_pre_right_point",
                        "direct_pre_left_point",
                        "direct_left_tracking",
                        "direct_right_tracking",
                        "direct_post_right_point",
                        "direct_post_left_point",
                    ]

                    evenOrder2 = [
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                        "natural_pre_left_point",
                        "natural_pre_right_point",
                        "natural_right_tracking",
                        "natural_left_tracking",
                        "natural_post_left_point",
                        "natural_post_right_point",
                        "direct_pre_left_point",
                        "direct_pre_right_point",
                        "direct_right_tracking",
                        "direct_left_tracking",
                        "direct_post_left_point",
                        "direct_post_right_point",
                    ]

                    # Order = 3 (Direct / Natural / Averted)
                    oddOrder3 = [
                        "direct_pre_right_point",
                        "direct_pre_left_point",
                        "direct_left_tracking",
                        "direct_right_tracking",
                        "direct_post_right_point",
                        "direct_post_left_point",
                        "natural_pre_right_point",
                        "natural_pre_left_point",
                        "natural_left_tracking",
                        "natural_right_tracking",
                        "natural_post_right_point",
                        "natural_post_left_point",
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                    ]

                    evenOrder3 = [
                        "direct_pre_left_point",
                        "direct_pre_right_point",
                        "direct_right_tracking",
                        "direct_left_tracking",
                        "direct_post_left_point",
                        "direct_post_right_point",
                        "natural_pre_left_point",
                        "natural_pre_right_point",
                        "natural_right_tracking",
                        "natural_left_tracking",
                        "natural_post_left_point",
                        "natural_post_right_point",
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                    ]

                    # Order = 4 (Direct/Averted/Natural)
                    oddOrder4 = [
                        "direct_pre_right_point",
                        "direct_pre_left_point",
                        "direct_left_tracking",
                        "direct_right_tracking",
                        "direct_post_right_point",
                        "direct_post_left_point",
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                        "natural_pre_right_point",
                        "natural_pre_left_point",
                        "natural_left_tracking",
                        "natural_right_tracking",
                        "natural_post_right_point",
                        "natural_post_left_point",
                    ]

                    evenOrder4 = [
                        "direct_pre_left_point",
                        "direct_pre_right_point",
                        "direct_right_tracking",
                        "direct_left_tracking",
                        "direct_post_left_point",
                        "direct_post_right_point",
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                        "natural_pre_left_point",
                        "natural_pre_right_point",
                        "natural_right_tracking",
                        "natural_left_tracking",
                        "natural_post_left_point",
                        "natural_post_right_point",
                    ]

                    # Order = 5 (Natural/Direct/Averted)
                    oddOrder5 = [
                        "natural_pre_right_point",
                        "natural_pre_left_point",
                        "natural_left_tracking",
                        "natural_right_tracking",
                        "natural_post_right_point",
                        "natural_post_left_point",
                        "direct_pre_right_point",
                        "direct_pre_left_point",
                        "direct_left_tracking",
                        "direct_right_tracking",
                        "direct_post_right_point",
                        "direct_post_left_point",
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                    ]

                    evenOrder5 = [
                        "natural_pre_left_point",
                        "natural_pre_right_point",
                        "natural_right_tracking",
                        "natural_left_tracking",
                        "natural_post_left_point",
                        "natural_post_right_point",
                        "direct_pre_left_point",
                        "direct_pre_right_point",
                        "direct_right_tracking",
                        "direct_left_tracking",
                        "direct_post_left_point",
                        "direct_post_right_point",
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                    ]

                    # Order = 6 (Natural/Averted/Direct)
                    oddOrder6 = [
                        "natural_pre_right_point",
                        "natural_pre_left_point",
                        "natural_left_tracking",
                        "natural_right_tracking",
                        "natural_post_right_point",
                        "natural_post_left_point",
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                        "direct_pre_right_point",
                        "direct_pre_left_point",
                        "direct_left_tracking",
                        "direct_right_tracking",
                        "direct_post_right_point",
                        "direct_post_left_point",
                    ]

                    evenOrder6 = [
                        "natural_pre_left_point",
                        "natural_pre_right_point",
                        "natural_right_tracking",
                        "natural_left_tracking",
                        "natural_post_left_point",
                        "natural_post_right_point",
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                        "direct_pre_left_point",
                        "direct_pre_right_point",
                        "direct_right_tracking",
                        "direct_left_tracking",
                        "direct_post_left_point",
                        "direct_post_right_point",
                    ]

                    # Populate all orders
                    listOfOrders = []
                    for i in tqdm(range(6)):
                        listOfOrders.append(eval("oddOrder" + str(i + 1)))
                        listOfOrders.append(eval("evenOrder" + str(i + 1)))

                    labelsequence = labelsequence - 1
                    chosenOrder = listOfOrders[labelsequence]

                    # Get the experimental markers' indices and extract the data
                    indicesofExperimentalMarkers = indicesOfMarkers[12:]

                    # Chunk the data based on opening and closing markers and get only the 16 channels data (columns)
                    chunkedData = []
                    for i in range(0, len(indicesofExperimentalMarkers), 2):

                        # Convert the data into numpy array type and microvolt (for some reason, the output of OpenBCI is not in microvolt)
                        chunkedData.append(
                            df.iloc[
                                indicesofExperimentalMarkers[
                                    i
                                ] : indicesofExperimentalMarkers[i + 1],
                                1:17,
                            ].to_numpy()
                            * 1e-6
                        )

                    # Create 16 channels montage 10-20 international standard
                    montage = mne.channels.make_standard_montage("standard_1020")

                    # Pick only 16 channels that are used in Cyton+Daisy OpenBCI
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

                    # Create info (which is used in MNE-Python)
                    info = mne.create_info(
                        ch_names=ch_names, sfreq=125, ch_types=ch_types
                    )
                    info.set_montage("standard_1020", match_case=False)

                    # Match pattern EEG-Sxx (xx = any number)
                    regex = r"\D{3}-S\d+"

                    # Create filenames for *.fif based on the sequence of labels above
                    extracted_file_name_4_experiment = []
                    for i in chosenOrder:
                        extracted_file_name = re.search(regex, fileName)
                        extracted_file_name_4_experiment.append(
                            fileName[
                                extracted_file_name.start() : extracted_file_name.end()
                            ]
                            + "-"
                            + i
                            + "_raw.fif"
                        )

                    # Save into *.fif files
                    for i, val in tqdm(
                        enumerate(chunkedData), desc="Saving process..."
                    ):
                        # Load data into MNE-Python
                        experiment_data_needs_label = mne.io.RawArray(
                            val.transpose(), info, verbose=False
                        )

                        # Define a folder where we want to save the experimental data
                        os.chdir(path_2_save_experimental_file)

                        # Save the data in MNE format
                        experiment_data_needs_label.save(
                            extracted_file_name_4_experiment[i], overwrite=True
                        )

        print(
            f"All experimental files have been saved in fif format in this path {path_2_save_experimental_file}"
        )

    ### Combine experimental hand data

    def combine_experimental_hand_data(self, path2data: str, path2storedata: str):
        """ * After the data of EEG has been extracted, it is separated between left and right hand data. \
            Due to that, we need to combine both data by using this function. During pre-training, participants need to point \
            their hands with right and left alternatively. It is 1 minute for each hand. Since there are two hands, \
            then it was 2 minutes for both hands. Similarly, during post-training, they need to do the same thing for both hands.

        :param path2data: path to separated raw EEG file (*fif).
        :type path2data: str
        :param path2storedata: path to save combined experimental data of EEG (*.fif).
        :type path2storedata: str
        :returns: EEG file
        :rtype: *.fif (mne)
        
        .. note:: * Each pair needs to point with the opposite hand. For example, if S1 points with right hand, \
                    then S2 needs to point with left hand.
                  * Odd subjects (1,3,5..) point with RIGHT-LEFT order.
                  * Even subjects(2,4,6..) point with LEFT-RIGHT order.
                  * The function has taken into consideration the above orders

                  * returns: 
                       * EEG file in :literal:`*.fif` format has the following name formatting :
                          * **SubjectNo_EyeCondition_TrainingCondition_HandCondition_raw.fif**

                       * In total, there are 6 files that will be resulted from each participant.\
                         e.g. :
                          #. S01-averted_post_left_right_point_combined_raw.fif
                          #. S01-averted_pre_right_left_point_combined_raw.fif
                          #. S01-direct_post_left_right_point_combined_raw.fif
                          #. S01-direct_pre_right_left_point_combined_raw.fif
                          #. S01-natural_post_left_right_point_combined_raw.fif
                          #. S01-natural_pre_right_left_point_combined_raw.fif

                       * Make sure the subject number that is written in file names begins with a leading zero, eg. 01, 02, 03.

        """

        # Change a working directory to where the extracted data is stored
        os.chdir(path2data)

        ### Odd subjects (1-9)

        begin = 0
        end = 9
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Pre-averted
            averted_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_pre_right_point_raw.fif",
                verbose=False,
            )
            averted_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False
            )
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(
                averted_pre_files_to_combine
            )
            combined_pre_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_averted_files.save(
                combined_pre_averted_files_label, overwrite=True
            )

            # Post-averted
            averted_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_post_right_point_raw.fif",
                verbose=False,
            )
            averted_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_post_left_point_raw.fif",
                verbose=False,
            )
            averted_post_files_to_combine = [
                averted_post_left_odd_subject,
                averted_post_right_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(
                averted_post_files_to_combine
            )
            combined_post_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )
            combined_post_averted_files.save(
                combined_post_averted_files_label, overwrite=True
            )

            #  Pre-directed
            direct_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-direct_pre_right_point_raw.fif", verbose=False
            )
            direct_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-direct_pre_left_point_raw.fif", verbose=False
            )
            direct_pre_files_to_combine = [
                direct_pre_right_odd_subject,
                direct_pre_left_odd_subject,
            ]
            combined_pre_direct_files = mne.concatenate_raws(
                direct_pre_files_to_combine
            )
            combined_pre_direct_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-direct_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_direct_files.save(
                combined_pre_direct_files_label, overwrite=True
            )

            #  Post-directed
            direct_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-direct_post_right_point_raw.fif",
                verbose=False,
            )
            direct_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-direct_post_left_point_raw.fif", verbose=False
            )
            direct_post_files_to_combine = [
                direct_post_left_odd_subject,
                direct_post_right_odd_subject,
            ]
            combined_post_direct_files = mne.concatenate_raws(
                direct_post_files_to_combine
            )
            combined_post_direct_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-direct_post_left_right_point_combined_raw.fif"
            )
            combined_post_direct_files.save(
                combined_post_direct_files_label, overwrite=True
            )

            #  Pre-natural
            natural_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-natural_pre_right_point_raw.fif",
                verbose=False,
            )
            natural_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-natural_pre_left_point_raw.fif", verbose=False
            )
            natural_pre_files_to_combine = [
                natural_pre_right_odd_subject,
                natural_pre_left_odd_subject,
            ]
            combined_pre_natural_files = mne.concatenate_raws(
                natural_pre_files_to_combine
            )
            combined_pre_natural_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-natural_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_natural_files.save(
                combined_pre_natural_files_label, overwrite=True
            )

            #  Post-natural
            natural_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-natural_post_right_point_raw.fif",
                verbose=False,
            )
            natural_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-natural_post_left_point_raw.fif",
                verbose=False,
            )
            natural_post_files_to_combine = [
                natural_post_left_odd_subject,
                natural_post_right_odd_subject,
            ]
            combined_post_natural_files = mne.concatenate_raws(
                natural_post_files_to_combine
            )
            combined_post_natural_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-natural_post_left_right_point_combined_raw.fif"
            )
            combined_post_natural_files.save(
                combined_post_natural_files_label, overwrite=True
            )

        print(
            "You files [Odd subjects (1-9)] have been combined, sir !. Just continue your coffee :)"
        )

        ## Even subjects 2 - 8

        begin = 0
        end = 8
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Pre-averted
            averted_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_pre_right_point_raw.fif",
                verbose=False,
            )
            averted_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_pre_left_point_raw.fif", verbose=False
            )
            averted_pre_files_to_combine = [
                averted_pre_left_odd_subject,
                averted_pre_right_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(
                averted_pre_files_to_combine
            )
            combined_pre_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-averted_pre_left_right_point_combined_raw.fif"
            )
            combined_pre_averted_files.save(
                combined_pre_averted_files_label, overwrite=True
            )

            # Post-averted
            averted_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_post_right_point_raw.fif",
                verbose=False,
            )
            averted_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_post_left_point_raw.fif",
                verbose=False,
            )
            averted_post_files_to_combine = [
                averted_post_right_odd_subject,
                averted_post_left_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(
                averted_post_files_to_combine
            )
            combined_post_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-averted_post_right_left_point_combined_raw.fif"
            )
            combined_post_averted_files.save(
                combined_post_averted_files_label, overwrite=True
            )

            #  Pre-directed
            direct_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-direct_pre_right_point_raw.fif", verbose=False
            )
            direct_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-direct_pre_left_point_raw.fif", verbose=False
            )
            direct_pre_files_to_combine = [
                direct_pre_left_odd_subject,
                direct_pre_right_odd_subject,
            ]
            combined_pre_direct_files = mne.concatenate_raws(
                direct_pre_files_to_combine
            )
            combined_pre_direct_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-direct_pre_left_right_point_combined_raw.fif"
            )
            combined_pre_direct_files.save(
                combined_pre_direct_files_label, overwrite=True
            )

            #  Post-directed
            direct_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-direct_post_right_point_raw.fif",
                verbose=False,
            )
            direct_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-direct_post_left_point_raw.fif", verbose=False
            )
            direct_post_files_to_combine = [
                direct_post_right_odd_subject,
                direct_post_left_odd_subject,
            ]
            combined_post_direct_files = mne.concatenate_raws(
                direct_post_files_to_combine
            )
            combined_post_direct_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-direct_post_right_left_point_combined_raw.fif"
            )
            combined_post_direct_files.save(
                combined_post_direct_files_label, overwrite=True
            )

            #  Pre-natural
            natural_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-natural_pre_right_point_raw.fif",
                verbose=False,
            )
            natural_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-natural_pre_left_point_raw.fif", verbose=False
            )
            natural_pre_files_to_combine = [
                natural_pre_left_odd_subject,
                natural_pre_right_odd_subject,
            ]
            combined_pre_natural_files = mne.concatenate_raws(
                natural_pre_files_to_combine
            )
            combined_pre_natural_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-natural_pre_left_right_point_combined_raw.fif"
            )
            combined_pre_natural_files.save(
                combined_pre_natural_files_label, overwrite=True
            )

            #  Post-natural
            natural_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-natural_post_right_point_raw.fif",
                verbose=False,
            )
            natural_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-natural_post_left_point_raw.fif",
                verbose=False,
            )
            natural_post_files_to_combine = [
                natural_post_right_odd_subject,
                natural_post_left_odd_subject,
            ]
            combined_post_natural_files = mne.concatenate_raws(
                natural_post_files_to_combine
            )
            combined_post_natural_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-natural_post_right_left_point_combined_raw.fif"
            )
            combined_post_natural_files.save(
                combined_post_natural_files_label, overwrite=True
            )

        print(
            "You files [Even subjects (2 - 8)] have been combined, sir !. Just continue your coffee :)"
        )

        #### Even subjects (10 and onwards, eg. 10, 12, 14, etc..)

        begin = 10
        end = 16
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Grab only file No. 10 and combine
            if i == 10:
                # Pre-averted
                averted_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_pre_right_point_raw.fif", verbose=False
                )
                averted_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_pre_left_point_raw.fif", verbose=False
                )
                averted_pre_files_to_combine = [
                    averted_pre_left_odd_subject,
                    averted_pre_right_odd_subject,
                ]
                combined_pre_averted_files = mne.concatenate_raws(
                    averted_pre_files_to_combine
                )
                combined_pre_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-averted_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_averted_files.save(
                    combined_pre_averted_files_label, overwrite=True
                )

                # Post-averted
                averted_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_post_right_point_raw.fif",
                    verbose=False,
                )
                averted_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_post_left_point_raw.fif", verbose=False
                )
                averted_post_files_to_combine = [
                    averted_post_right_odd_subject,
                    averted_post_left_odd_subject,
                ]
                combined_post_averted_files = mne.concatenate_raws(
                    averted_post_files_to_combine
                )
                combined_post_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-averted_post_right_left_point_combined_raw.fif"
                )
                combined_post_averted_files.save(
                    combined_post_averted_files_label, overwrite=True
                )

                # Pre-directed
                direct_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-direct_pre_right_point_raw.fif", verbose=False
                )
                direct_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-direct_pre_left_point_raw.fif", verbose=False
                )
                direct_pre_files_to_combine = [
                    direct_pre_left_odd_subject,
                    direct_pre_right_odd_subject,
                ]
                combined_pre_direct_files = mne.concatenate_raws(
                    direct_pre_files_to_combine
                )
                combined_pre_direct_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-direct_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_direct_files.save(
                    combined_pre_direct_files_label, overwrite=True
                )

                # Post-directed
                direct_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-direct_post_right_point_raw.fif", verbose=False
                )
                direct_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-direct_post_left_point_raw.fif", verbose=False
                )
                direct_post_files_to_combine = [
                    direct_post_right_odd_subject,
                    direct_post_left_odd_subject,
                ]
                combined_post_direct_files = mne.concatenate_raws(
                    direct_post_files_to_combine
                )
                combined_post_direct_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-direct_post_right_left_point_combined_raw.fif"
                )
                combined_post_direct_files.save(
                    combined_post_direct_files_label, overwrite=True
                )

                # Pre-natural
                natural_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-natural_pre_right_point_raw.fif", verbose=False
                )
                natural_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-natural_pre_left_point_raw.fif", verbose=False
                )
                natural_pre_files_to_combine = [
                    natural_pre_left_odd_subject,
                    natural_pre_right_odd_subject,
                ]
                combined_pre_natural_files = mne.concatenate_raws(
                    natural_pre_files_to_combine
                )
                combined_pre_natural_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-natural_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_natural_files.save(
                    combined_pre_natural_files_label, overwrite=True
                )

                # Post-natural
                natural_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-natural_post_right_point_raw.fif",
                    verbose=False,
                )
                natural_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-natural_post_left_point_raw.fif", verbose=False
                )
                natural_post_files_to_combine = [
                    natural_post_right_odd_subject,
                    natural_post_left_odd_subject,
                ]
                combined_post_natural_files = mne.concatenate_raws(
                    natural_post_files_to_combine
                )
                combined_post_natural_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-natural_post_right_left_point_combined_raw.fif"
                )
                combined_post_natural_files.save(
                    combined_post_natural_files_label, overwrite=True
                )

            # Combine file no.12, 14, etc..
            if i + 2 > 10:
                # Pre-averted
                averted_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_pre_right_point_raw.fif",
                    verbose=False,
                )
                averted_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_pre_left_point_raw.fif",
                    verbose=False,
                )
                averted_pre_files_to_combine = [
                    averted_pre_left_odd_subject,
                    averted_pre_right_odd_subject,
                ]
                combined_pre_averted_files = mne.concatenate_raws(
                    averted_pre_files_to_combine
                )
                combined_pre_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-averted_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_averted_files.save(
                    combined_pre_averted_files_label, overwrite=True
                )

                # Post-averted
                averted_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_post_right_point_raw.fif",
                    verbose=False,
                )
                averted_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_post_left_point_raw.fif",
                    verbose=False,
                )
                averted_post_files_to_combine = [
                    averted_post_right_odd_subject,
                    averted_post_left_odd_subject,
                ]
                combined_post_averted_files = mne.concatenate_raws(
                    averted_post_files_to_combine
                )
                combined_post_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-averted_post_right_left_point_combined_raw.fif"
                )
                combined_post_averted_files.save(
                    combined_post_averted_files_label, overwrite=True
                )

                #  Pre-directed
                direct_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-direct_pre_right_point_raw.fif",
                    verbose=False,
                )
                direct_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-direct_pre_left_point_raw.fif",
                    verbose=False,
                )
                direct_pre_files_to_combine = [
                    direct_pre_left_odd_subject,
                    direct_pre_right_odd_subject,
                ]
                combined_pre_direct_files = mne.concatenate_raws(
                    direct_pre_files_to_combine
                )
                combined_pre_direct_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-direct_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_direct_files.save(
                    combined_pre_direct_files_label, overwrite=True
                )

                #  Post-directed
                direct_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-direct_post_right_point_raw.fif",
                    verbose=False,
                )
                direct_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-direct_post_left_point_raw.fif",
                    verbose=False,
                )
                direct_post_files_to_combine = [
                    direct_post_right_odd_subject,
                    direct_post_left_odd_subject,
                ]
                combined_post_direct_files = mne.concatenate_raws(
                    direct_post_files_to_combine
                )
                combined_post_direct_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-direct_post_right_left_point_combined_raw.fif"
                )
                combined_post_direct_files.save(
                    combined_post_direct_files_label, overwrite=True
                )

                #  Pre-natural
                natural_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-natural_pre_right_point_raw.fif",
                    verbose=False,
                )
                natural_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-natural_pre_left_point_raw.fif",
                    verbose=False,
                )
                natural_pre_files_to_combine = [
                    natural_pre_left_odd_subject,
                    natural_pre_right_odd_subject,
                ]
                combined_pre_natural_files = mne.concatenate_raws(
                    natural_pre_files_to_combine
                )
                combined_pre_natural_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-natural_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_natural_files.save(
                    combined_pre_natural_files_label, overwrite=True
                )

                #  Post-natural
                natural_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-natural_post_right_point_raw.fif",
                    verbose=False,
                )
                natural_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-natural_post_left_point_raw.fif",
                    verbose=False,
                )
                natural_post_files_to_combine = [
                    natural_post_right_odd_subject,
                    natural_post_left_odd_subject,
                ]
                combined_post_natural_files = mne.concatenate_raws(
                    natural_post_files_to_combine
                )
                combined_post_natural_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-natural_post_right_left_point_combined_raw.fif"
                )
                combined_post_natural_files.save(
                    combined_post_natural_files_label, overwrite=True
                )

        print(
            "You files [Even subjects (10 and onwards, eg. 10, 12, 14, etc..)] have been combined, sir !. Just continue your coffee :)"
        )

        #### Odd subjects (11 and onwards, eg. 11, 13, 15, etc...)

        begin = 10
        end = 16
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Pre-averted
            averted_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False
            )
            averted_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False
            )
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(
                averted_pre_files_to_combine
            )
            combined_pre_averted_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_averted_files.save(
                combined_pre_averted_files_label, overwrite=True
            )

            # Post-averted
            averted_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_post_right_point_raw.fif",
                verbose=False,
            )
            averted_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False
            )
            averted_post_files_to_combine = [
                averted_post_left_odd_subject,
                averted_post_right_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(
                averted_post_files_to_combine
            )
            combined_post_averted_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )
            combined_post_averted_files.save(
                combined_post_averted_files_label, overwrite=True
            )

            #  Pre-directed
            direct_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-direct_pre_right_point_raw.fif", verbose=False
            )
            direct_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-direct_pre_left_point_raw.fif", verbose=False
            )
            direct_pre_files_to_combine = [
                direct_pre_right_odd_subject,
                direct_pre_left_odd_subject,
            ]
            combined_pre_direct_files = mne.concatenate_raws(
                direct_pre_files_to_combine
            )
            combined_pre_direct_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-direct_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_direct_files.save(
                combined_pre_direct_files_label, overwrite=True
            )

            #  Post-directed
            direct_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-direct_post_right_point_raw.fif", verbose=False
            )
            direct_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-direct_post_left_point_raw.fif", verbose=False
            )
            direct_post_files_to_combine = [
                direct_post_left_odd_subject,
                direct_post_right_odd_subject,
            ]
            combined_post_direct_files = mne.concatenate_raws(
                direct_post_files_to_combine
            )
            combined_post_direct_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-direct_post_left_right_point_combined_raw.fif"
            )
            combined_post_direct_files.save(
                combined_post_direct_files_label, overwrite=True
            )

            #  Pre-natural
            natural_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-natural_pre_right_point_raw.fif", verbose=False
            )
            natural_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-natural_pre_left_point_raw.fif", verbose=False
            )
            natural_pre_files_to_combine = [
                natural_pre_right_odd_subject,
                natural_pre_left_odd_subject,
            ]
            combined_pre_natural_files = mne.concatenate_raws(
                natural_pre_files_to_combine
            )
            combined_pre_natural_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-natural_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_natural_files.save(
                combined_pre_natural_files_label, overwrite=True
            )

            #  Post-natural
            natural_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-natural_post_right_point_raw.fif",
                verbose=False,
            )
            natural_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-natural_post_left_point_raw.fif", verbose=False
            )
            natural_post_files_to_combine = [
                natural_post_left_odd_subject,
                natural_post_right_odd_subject,
            ]
            combined_post_natural_files = mne.concatenate_raws(
                natural_post_files_to_combine
            )
            combined_post_natural_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-natural_post_left_right_point_combined_raw.fif"
            )
            combined_post_natural_files.save(
                combined_post_natural_files_label, overwrite=True
            )

        print(
            "You files [Odd subjects (11 and onwards, eg. 11, 13, 15, etc...)] have been combined, sir !. Just continue your coffee :)"
        )

        ### Combine baseline data

    def combine_baseline_hand_data(self, path2data: str, path2storedata: str):

        """ * After the data of EEG has been extracted, it is separated between left and right hand data. \
            Due to that, we need to combine both data by using this function. During pre-training, participants need to point \
            their hands with right and left alternatively. It is 1 minute for each hand. Since there are two hands, \
            then it was 2 minutes for both hands. Similarly, during post-training, they need to do the same thing for both hands.

        :param path2data: path to separated raw EEG file (*fif).
        :type path2data: str
        :param path2storedata: path to save combined baseline data of EEG (*.fif).
        :type path2storedata: str
        :returns: EEG file
        :rtype: *.fif (mne)
                
        .. note:: * Each pair needs to point with the opposite hand. For example, if S1 points with right hand.
                    then S2 needs to point with left hand.

                  * Odd subjects (1,3,5..) point with RIGHT-LEFT order.
                  * Even subjects(2,4,6..) point with LEFT-RIGHT order.
                  * The function has taken into consideration the above orders.
                  * Make sure the subject number that is written in file names begins with a leading zero, eg. 01, 02, 03.

                  * parameters: path2data
                      * EEG file in :literal:`*.fif` format has the following name formatting :
                        * EEG-SubjectNo-EyeCondition_HandCondition_raw.fif
                            * **EEG-S01-averted_left_point_raw.fif**
                        * EyeCondition is ONLY AVERTED.
                        * **note** :  * This function combines only hand pointing NOT TRACKING.\
                                      Because we are interested in pre vs post training.\
                                      Hand tracking is in training condition.

                                      * The eye condition is ONLY averted.\
                                      Basically, each participant sees only white screen. So whatever eye condition\
                                      does not really matter. For the sake of coherence. We just put it averted\
                                      During recording, the eye condition is set to averted in UNITY. 


                  * returns: EEG file in :literal:`*.fif` format has the following name formatting.\
                             2 files will be resulted from each participant:
                           * EEG-SubjectNo_EyeCondition_TrainingCondition_HandCondition_raw.fif
                               #. S01-averted_post_left_right_point_combined_raw.fif
                               #. S01-averted_pre_right_left_point_combined_raw.fif   
        """

        # Change a working directory to a folder where extracted baseline data is stored
        os.chdir(path2data)

        ### Odd subjects (1-9)
        begin = 0
        end = 9
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Pre-averted
            averted_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_pre_right_point_raw.fif",
                verbose=False,
            )
            averted_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False
            )
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(
                averted_pre_files_to_combine
            )
            combined_pre_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_averted_files.save(
                combined_pre_averted_files_label, overwrite=True
            )

            # Post-averted
            averted_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_post_right_point_raw.fif",
                verbose=False,
            )
            averted_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 1) + "-averted_post_left_point_raw.fif",
                verbose=False,
            )
            averted_post_files_to_combine = [
                averted_post_left_odd_subject,
                averted_post_right_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(
                averted_post_files_to_combine
            )
            combined_post_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )
            combined_post_averted_files.save(
                combined_post_averted_files_label, overwrite=True
            )

        print(
            "You files[Odd subjects (1-9)] have been combined, sir !. Just continue your coffee :)"
        )

        ### Even subjects (2-8)
        begin = 0
        end = 8
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Pre-averted
            averted_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_pre_right_point_raw.fif",
                verbose=False,
            )
            averted_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_pre_left_point_raw.fif", verbose=False
            )
            averted_pre_files_to_combine = [
                averted_pre_left_odd_subject,
                averted_pre_right_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(
                averted_pre_files_to_combine
            )
            combined_pre_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-averted_pre_left_right_point_combined_raw.fif"
            )
            combined_pre_averted_files.save(
                combined_pre_averted_files_label, overwrite=True
            )

            # Post-averted
            averted_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_post_right_point_raw.fif",
                verbose=False,
            )
            averted_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S0" + str(i + 2) + "-averted_post_left_point_raw.fif",
                verbose=False,
            )
            averted_post_files_to_combine = [
                averted_post_right_odd_subject,
                averted_post_left_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(
                averted_post_files_to_combine
            )
            combined_post_averted_files_label = (
                path2storedata
                + "S0"
                + str(i + 2)
                + "-averted_post_right_left_point_combined_raw.fif"
            )
            combined_post_averted_files.save(
                combined_post_averted_files_label, overwrite=True
            )

        print(
            "You files [Even subjects (2-8)] have been combined, sir !. Just continue your coffee :)"
        )

        ### Even subjects (10 and onwards, eg. 10, 12, 14, etc..)
        begin = 10
        end = 16
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Grab only file No. 10 and combine
            if i == 10:
                # Pre-averted
                averted_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_pre_right_point_raw.fif", verbose=False
                )
                averted_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_pre_left_point_raw.fif", verbose=False
                )
                averted_pre_files_to_combine = [
                    averted_pre_left_odd_subject,
                    averted_pre_right_odd_subject,
                ]
                combined_pre_averted_files = mne.concatenate_raws(
                    averted_pre_files_to_combine
                )
                combined_pre_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-averted_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_averted_files.save(
                    combined_pre_averted_files_label, overwrite=True
                )

                # Post-averted
                averted_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_post_right_point_raw.fif",
                    verbose=False,
                )
                averted_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i) + "-averted_post_left_point_raw.fif", verbose=False
                )
                averted_post_files_to_combine = [
                    averted_post_right_odd_subject,
                    averted_post_left_odd_subject,
                ]
                combined_post_averted_files = mne.concatenate_raws(
                    averted_post_files_to_combine
                )
                combined_post_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i)
                    + "-averted_post_right_left_point_combined_raw.fif"
                )
                combined_post_averted_files.save(
                    combined_post_averted_files_label, overwrite=True
                )

            # Combine file no.12, 14, etc..
            if i + 2 > 10:
                # Pre-averted
                averted_pre_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_pre_right_point_raw.fif",
                    verbose=False,
                )
                averted_pre_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_pre_left_point_raw.fif",
                    verbose=False,
                )
                averted_pre_files_to_combine = [
                    averted_pre_left_odd_subject,
                    averted_pre_right_odd_subject,
                ]
                combined_pre_averted_files = mne.concatenate_raws(
                    averted_pre_files_to_combine
                )
                combined_pre_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-averted_pre_left_right_point_combined_raw.fif"
                )
                combined_pre_averted_files.save(
                    combined_pre_averted_files_label, overwrite=True
                )

                # Post-averted
                averted_post_right_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_post_right_point_raw.fif",
                    verbose=False,
                )
                averted_post_left_odd_subject = mne.io.read_raw_fif(
                    "EEG-S" + str(i + 2) + "-averted_post_left_point_raw.fif",
                    verbose=False,
                )
                averted_post_files_to_combine = [
                    averted_post_right_odd_subject,
                    averted_post_left_odd_subject,
                ]
                combined_post_averted_files = mne.concatenate_raws(
                    averted_post_files_to_combine
                )
                combined_post_averted_files_label = (
                    path2storedata
                    + "S"
                    + str(i + 2)
                    + "-averted_post_right_left_point_combined_raw.fif"
                )
                combined_post_averted_files.save(
                    combined_post_averted_files_label, overwrite=True
                )

        print(
            "You files [Even subjects (10 and onwards, eg. 10, 12..] have been combined, sir !. Just continue your coffee :)"
        )

        #### Odd subjects (11 and onwards, eg. 11, 13, 15, etc...)

        begin = 10
        end = 16
        step = 2

        for i in tqdm(
            range(begin, end, step), desc="Just relax and drink your coffee.."
        ):

            # Pre-averted
            averted_pre_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False
            )
            averted_pre_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False
            )
            averted_pre_files_to_combine = [
                averted_pre_right_odd_subject,
                averted_pre_left_odd_subject,
            ]
            combined_pre_averted_files = mne.concatenate_raws(
                averted_pre_files_to_combine
            )
            combined_pre_averted_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-averted_pre_right_left_point_combined_raw.fif"
            )
            combined_pre_averted_files.save(
                combined_pre_averted_files_label, overwrite=True
            )

            # Post-averted
            averted_post_right_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_post_right_point_raw.fif",
                verbose=False,
            )
            averted_post_left_odd_subject = mne.io.read_raw_fif(
                "EEG-S" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False
            )
            averted_post_files_to_combine = [
                averted_post_left_odd_subject,
                averted_post_right_odd_subject,
            ]
            combined_post_averted_files = mne.concatenate_raws(
                averted_post_files_to_combine
            )
            combined_post_averted_files_label = (
                path2storedata
                + "S"
                + str(i + 1)
                + "-averted_post_left_right_point_combined_raw.fif"
            )
            combined_post_averted_files.save(
                combined_post_averted_files_label, overwrite=True
            )

        print(
            "You files [Odd subjects (11 and onwards, eg. 11, 13..] have been combined, sir !. Just continue your coffee :)"
        )
