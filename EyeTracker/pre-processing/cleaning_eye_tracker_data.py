# %% Import packages
import chunk
import os
import re
import warnings

import mne
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% Extract baseline data


def extract_baseline_eye_data(
    path_2_csv_files: str,
    path_2_save_baseline_file: str,
    labelsequence: int = 1,
    bad_files: list = [],
):
    """
    Extract baseline data from raw EEG file (*.csv) that was obtained from hyperscanning2-redesign experiment \n

    Arguments :
        - path_2_csv_files (str) : path to raw EEG file \n
        - path_2_save_baseline_file (str) : path to save extracted baseline file for each condition \n
        - labelsequence (int) : order of pre-defined label sequence, 1 (averted) is default \n
        - bad_files (list) (optional) : file name of raw EEG file, e.g., EEG-S8.csv, that wants to be skipped to process

    Return :
        Extracted *.fif (MNE-Python) file for each condition of hand (finger pointing and tracking).
        There are 6 files in total for each participant.
    """

    list_file_names = []
    full_path_2_each_file = []
    # bad_files=["EEG-S8.csv"] # Hard coded for now because bad EEG file is inside a folder of EEG. Remove this later

    for (root, dirs, file) in os.walk(path_2_csv_files):
        for f in file:

            if ".csv" in f:

                if f in bad_files:

                    # Skip the bad file to be processed
                    print(f"Skipped bad file : {f}")
                    continue

                else:
                    # Populate all file names only
                    list_file_names.append(f)
                    list_file_names.sort()

                    # Populate all full paths of each filename
                    full_path_2_each_file.append(os.path.join(root, f))
                    full_path_2_each_file.sort()

        # Iterate all file names

        for i in tqdm(range(len(full_path_2_each_file)), desc="In progress"):

            try:
                labelsequence = int(labelsequence)

            except IOError as err_filename:
                print(
                    "The format of file name is not correct or file doesn't exist \nThe format must be 'EEG-Sx.csv' , x=subject number "
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
                    fileName = full_path_2_each_file[i]
                    print("Processing file : " + list_file_names[i])

                    # Read each file by using pandas
                    df = pd.read_csv(fileName, delimiter=",")
                    # Define columns for raw csv file
                    # df.columns = ['Index', 'FP1', 'FP2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1',
                    #                 'O2', 'X1', 'X2', 'X3', 'X4', 'X5',
                    #                 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'Marker']

                    # Replace all markers of "BEGIN*" and "END*" with 9999999
                    df["UnixTimeStamp"] = df.UnixTimeStamp.apply(
                        lambda x: "9999999" if "BEGIN" in x else x
                    )
                    df["UnixTimeStamp"] = df.UnixTimeStamp.apply(
                        lambda x: "9999999" if "END" in x else x
                    )

                    # Turn the UnixTimeStamp column into a list (we need the marker later on)
                    markers = df["UnixTimeStamp"].tolist()

                    #   Find all experimental markers and print them out.
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

                    # Create a list of labels for baseline data. We used only averted eye condition in UNITY.
                    # It actually does not matter for different eye condition because participant only sees a white screen during the baseline condition)

                    # Order = 1 (Averted) Odd subject no. For example, 1, 3, 5, etc.
                    oddOrder1 = [
                        "averted_pre_right_point",
                        "averted_pre_left_point",
                        "averted_left_tracking",
                        "averted_right_tracking",
                        "averted_post_right_point",
                        "averted_post_left_point",
                    ]

                    # Order = 1 (Averted) Even subject no. For example, 2, 4, 6, etc.
                    evenOrder1 = [
                        "averted_pre_left_point",
                        "averted_pre_right_point",
                        "averted_right_tracking",
                        "averted_left_tracking",
                        "averted_post_left_point",
                        "averted_post_right_point",
                    ]

                    # Put all labels into a list for baseline data
                    listOfOrders = []
                    listOfOrders.append(oddOrder1)
                    listOfOrders.append(evenOrder1)

                    # Number that is used to take the label (oddOrder1 atau evenOrder1)
                    i_label_taker = 0

                    if i % 2 == 0:

                        # Even number
                        i_label_taker = 0

                    else:

                        # Odd number
                        i_label_taker = 1

                    chosenOrder = listOfOrders[i_label_taker]

                    # Get the first 12 markers' indices and extract the data
                    indicesofBaselineMarkers = indicesOfMarkers[:13]

                    # Get the 1st and 12th index and chunk dataframe based on those indices, and convert it into numpy array
                    # For some data, it can be 13 markers after being extracted because when we combined the data the markers of beginning are right after the closing marker

                    # Chunk the data based on opening and closing markers and get only the 16 channels data (columns)
                    chunkedData = []
                    for i in range(0, 12, 2):

                        chunkedData.append(
                            df.iloc[
                                indicesofBaselineMarkers[i] : indicesofBaselineMarkers[
                                    i + 1
                                ],
                                1:17,
                            ].to_numpy()
                            * 1e-6
                        )

                    # Load each baseline file into MNE Python (averted eye condition only for baseline)

                    # # Create 16 channels montage 10-20 international standard
                    # montage = mne.channels.make_standard_montage('standard_1020')

                    # # Pick only 16 channels that are used in Cyton+Daisy OpenBCI
                    # # Create info
                    # ch_names = ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
                    # ch_types = ['eeg'] * 16
                    # info = mne.create_info(
                    #     ch_names=ch_names,
                    #     sfreq=125,
                    #     ch_types=ch_types)
                    # info.set_montage('standard_1020', match_case=False)

                    # Match pattern EEG-Sx (x = any number)
                    regex = r"\D{10}-S\d+"

                    # Create filename that will be used for each condition. There are 6 conditions. See oddOrder1 or evenOrder1
                    extracted_file_name_4_baseline = []
                    for i in chosenOrder:
                        extracted_file_name = re.search(regex, fileName)
                        extracted_file_name_4_baseline.append(
                            fileName[
                                extracted_file_name.start() : extracted_file_name.end()
                            ]
                            + "-"
                            + i
                            + "_raw.fif"
                        )

                    # Save the chunkedData into a separate csv file
                    for i, val in tqdm(
                        enumerate(chunkedData), desc="Saving process..."
                    ):

                        # chunkedData[i].to_csv(os.chdir(path_2_save_baseline_file), extracted_file_name_4_baseline[i])
                        os.chdir(path_2_save_baseline_file)
                        chunkedData.tofile(extracted_file_name_4_baseline[i], sep=",")

                    # for i, val in tqdm(enumerate(chunkedData), desc = "Saving process..."):
                    #     # Load data into MNE-Python
                    #     baseline_data_needs_label = mne.io.RawArray(val.transpose(), info, verbose=False)
                    #     # Define a folder where we want to save the baseline data
                    #     os.chdir(path_2_save_baseline_file)
                    #     # Save the data in MNE format
                    #     baseline_data_needs_label.save(extracted_file_name_4_baseline[i], overwrite=True)

    print(
        f"All baseline files have been saved in fif format in this path {path_2_save_baseline_file}"
    )


# %% Testing extract_baseline_eye_data function
path_2_csv_files = "/hpc/igum002/codes/Hyperscanning2-redesign/EyeTracker/data/"
path_2_save_files = (
    "/hpc/igum002/codes/Hyperscanning2-redesign/EyeTracker/data/raw_baseline_eye_data/"
)

extract_baseline_eye_data(path_2_csv_files, path_2_save_files)


# %% Extract experimental data


def extract_experimental_data(
    path_2_csv_files: str,
    path_2_save_experimental_file: str,
    labelsequence_experiment: list,
    bad_files: list = [],
):
    """
    Extract experimental data from raw EEG file (*.csv) that was obtained from hyperscanning2-redesign experiment \n

    Arguments :
        - path_2_csv_files (str) : path to raw EEG file \n
        - path_2_save_experimental_file (str) : path to save extracted baseline file for each condition \n
        - labelsequence (int) : order of pre-defined label sequence \n
        - bad_files (list) (optional) : file name(s) of raw EEG data, e.g., EEG-S8.csv, that wants to be skipped to process

    Return :
        Extracted *.fif (MNE-Python) file for each condition of hand (finger pointing and tracking).
        There are 18 files in total for each participant.
    """

    list_file_names = []
    full_path_2_each_file = []
    # bad_files=["EEG-S8.csv"] # Hard coded for now because bad EEG file is inside a folder of EEG. Remove this later

    for (root, dirs, file) in os.walk(path_2_csv_files):
        for f in file:

            if ".csv" in f:

                if f in bad_files:

                    # Skip the bad file to be processed
                    print(f"Skipped bad file : {f}")
                    continue

                else:
                    # Populate all file names only
                    list_file_names.append(f)
                    list_file_names.sort()

                    # Populate all full paths of each filename
                    full_path_2_each_file.append(os.path.join(root, f))
                    full_path_2_each_file.sort()

        # Iterate all file names

        for i in tqdm(range(len(full_path_2_each_file)), desc="In progress"):

            try:
                labelsequence = int(labelsequence_experiment[i])

            except IOError as err_filename:
                print(
                    "The format of file name is not correct or file doesn't exist \nThe format must be 'EEG-Sx.csv' , x=subject number "
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
                    fileName = full_path_2_each_file[i]
                    print("Processing file : " + list_file_names[i])

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

                    # Find all experimental markers and print them out.
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

                    chosenOrder = listOfOrders[labelsequence - 1]

                    # Chunk the data based on opening and closing markers and get only the 16 channels data (columns)
                    chunkedData = []
                    for i in range(0, 36, 2):

                        # Change into numpy and convert it from uV (microvolts) / nV to V (volts)
                        chunkedData.append(
                            df.iloc[
                                indicesOfMarkers[i] : indicesOfMarkers[i + 1] + 1, 1:17
                            ].to_numpy()
                            * 1e-6
                        )

                    # Get 12 markers' indices and extract experimental data
                    indicesofBaselineMarkers = indicesOfMarkers[13:]

                    # Create 16 channels montage 10-20 international standard
                    montage = mne.channels.make_standard_montage("standard_1020")

                    # Pick only 16 channels that are used in Cyton+Daisy OpenBCI
                    # Create info
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

                    # Match pattern EEG-Sx (x = any number)
                    regex = r"\D{3}-S\d+"

                    # Create filename that will be used for each condition. There are 18 conditions for each participant.
                    extracted_file_name_4_baseline = []
                    for i in chosenOrder:
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
                        os.chdir(path_2_save_experimental_file)
                        # Save the data in MNE format
                        baseline_data_needs_label.save(
                            extracted_file_name_4_baseline[i], overwrite=True
                        )

    print(
        f"All experimental files have been saved in fif format in this path {path_2_save_experimental_file}"
    )
