import mne
import pandas as pd
from tqdm import tqdm
import warnings
import os
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_baseline_eeg_data(
    path_2_csv_files, path_2_save_baseline_file, labelsequence=1, bad_files=[]
):
    """

    Objective :
        Extract baseline data from raw EEG data which is in csv format. \n
        Every raw csv file of EEG data must contain 48 markers in total (opening & closing). \n
        Basically, there are 24 markers. However, Baseline data is in the first 6 markers. \n
        (12 markers if including opening & closing markers). \n


    Parameters :
        - path_2_csv_files : path to raw EEG file (csv format). \n
        - path_2_save_baseline_file : path to save extracted baseline data of EEG (output file in *.fif format). \n
        - labelsequence (opt) : order of label sequence, in this case is 1 by default. \n
        - bad_files (opt) : raw EEG file(s) that want to be skipped to be processed by the script.

    Output :
        EEG file in *.fif format with the name formatting : Subject no_EyeCondition__TrainingCondition_HandCondition_raw.fif
        For instance, "EEG-S01-averted_left_tracking_raw". There are 6 files in total for each participant.

        REMEMBER : All resulted files will be in "AVERTED" condition since the baseline condition is in AVERTED condition.

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

                    #%% Load the data
                    fileName = list_file_names[i]
                    print("Processing file : " + list_file_names[i])

                    # Change working directory to which raw EEG files are stored
                    os.chdir(path_2_csv_files)

                    # Read each file by using pandas
                    df = pd.read_csv(fileName, delimiter=",")
                    # %% Define columns for raw csv file
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

                    # %% Get the first 12 markers' indices and extract the data
                    indicesofBaselineMarkers = indicesOfMarkers[:13]

                    # Get the first 12 markers and chunk dataframe based on those indices, and convert it into numpy array
                    # For some data, it can be 13 markers after being extracted because when we combined the data the markers of beginning are
                    # right after the closing marker

                    # %% Chunk the data based on opening and closing markers and get only the 16 channels data (columns)
                    chunkedData = []
                    for i in range(0, 12, 2):

                        # Convert the data into numpy array type and microvolt (for some reason, the output of OpenBCI is not in microvolt)
                        chunkedData.append(
                            df.iloc[
                                indicesofBaselineMarkers[i] : indicesofBaselineMarkers[
                                    i + 1
                                ],
                                1:17,
                            ].to_numpy()
                            * 1e-6
                        )

                    # Create 16 channels montage 10-20 international standard
                    montage = mne.channels.make_standard_montage("standard_1020")

                    # Pick only 16 channels that are used in Cyton+Daisy OpenBCI
                    # % Create info (which is used in MNE-Python)
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

                    # %% Create filenames for *.fif based on the sequence of labels above
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

                    #%% Save into *.fif files
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
    path_2_csv_files,
    path_2_save_experimental_file,
    labelsequence_experiment: list,
    bad_files=[],
):

    """

    Objective :
        Extract experimental data from raw EEG data which is in csv format. \n
        Every raw csv file of EEG data must contain 48 markers in total (opening & closing). \n
        Basically, there are 24 markers. However, experimental data is from marker 7 to 24. \n
        (36 markers if including opening & closing markers). \n


    Parameters :
        - path_2_csv_files (str) : path to raw EEG file (csv format). \n
        - path_2_save_experimental_file (str) : path to save extracted experimental data of EEG (output file in *.fif format). \n
        - labelsequence_experiment (list) : order of label sequence. \n
        - bad_files (list) (opt) : raw EEG file(s) that want to be skipped to be processed by the script.

    Output :
        EEG file in *.fif format with the name formatting : Subject no_EyeCondition__TrainingCondition_HandCondition_raw.fif
        For instance, "EEG-S01-averted_left_tracking_raw". In total, there will be 18 files for each participant.

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

                #%% Load the data
                fileName = list_file_names[i]
                print("Processing file : " + list_file_names[i])

                # Change working directory to which raw EEG files are stored
                os.chdir(path_2_csv_files)

                # Read each file by using pandas
                df = pd.read_csv(fileName, delimiter=",")
                # %% Define columns for raw csv file
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
                info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)
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
                for i, val in tqdm(enumerate(chunkedData), desc="Saving process..."):
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
