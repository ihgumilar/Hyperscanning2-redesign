# %% Import packages
import chunk
import mne
import pandas as pd
from tqdm import tqdm
import warnings
import os
import re
import numbers

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% Extract baseline data


def extract_baseline_eye_data(
    path_2_csv_files: str,
    path_2_save_baseline_file: str,
    bad_files: list = [],
    labelsequence: int = 1,
):
    """
    Extract baseline data from raw Eye Tracker file (*.csv) that was obtained from hyperscanning2-redesign experiment \n

    Arguments :
        - path_2_csv_files (str) : path to raw Eye tracker file \n
        - path_2_save_baseline_file (str) : path to save extracted baseline file for each condition \n
        - bad_files (list) (optional) : file name of raw EyeTracker file, e.g., EyeTracker-S8.csv, that wants to be skipped to process
        - labelsequence (int) : order of pre-defined label sequence, 1 (averted) is default \n

    Return :
        Extracted *.csv file for each condition of hand (finger pointing and tracking).
        There are 6 files in total for each participant.

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

        # Iterate all file names

        for i in tqdm(range(len(list_file_names)), desc="In progress"):

            try:
                labelsequence = int(labelsequence)

            except IOError as err_filename:
                print(
                    "The format of file name is not correct or file doesn't exist \nThe format must be 'EyeTracker-Sx.csv' , x=subject number "
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

                    # Change to a folder where original CSV files are stored
                    os.chdir(path_2_csv_files)

                    print("Processing file : " + list_file_names[i])

                    # Read each file by using pandas
                    df = pd.read_csv(fileName, delimiter=",")

                    df["UnixTimeStamp"] = df.UnixTimeStamp.apply(
                        lambda x: "9999999" if "BEGIN" in x else x
                    )
                    df["UnixTimeStamp"] = df.UnixTimeStamp.apply(
                        lambda x: "9999999" if "END" in x else x
                    )

                    # Turn the UnixTimeStamp column into a list (we need the marker later on)
                    markers = df["UnixTimeStamp"].tolist()

                    # Convert string value to integer number
                    markers = list(map(int, markers))

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
                    indicesofBaselineMarkers = indicesOfMarkers[:12]

                    # Chunk the data based on opening and closing markers
                    chunkedData = []
                    for i in range(0, len(indicesofBaselineMarkers), 2):

                        chunkedData.append(
                            df.iloc[
                                indicesofBaselineMarkers[i] : indicesofBaselineMarkers[
                                    i + 1
                                ],
                                :,
                            ]
                        )

                    # Match pattern EyeTracker-Sx (x = any number)
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
                            + "_raw.csv"
                        )

                    # Save the chunkedData into a separate csv file
                    for i, val in tqdm(
                        enumerate(chunkedData), desc="Saving process..."
                    ):

                        # Convert array into dataframe
                        df_chunkedData = pd.DataFrame(val)

                        # Save dataframe into csv
                        os.chdir(path_2_save_baseline_file)
                        df_chunkedData.to_csv(
                            extracted_file_name_4_baseline[i], sep=(",")
                        )

    print(
        f"All baseline files of eye data have been saved in csv format in this path {path_2_save_baseline_file}"
    )


# %%Extract experimental data


def extract_experiment_eye_data(
    path_2_csv_files: str,
    path_2_save_experimental_file: str,
    labelsequence_experiment: list,
    bad_files: list = [],
):
    """
    Extract experimental data from raw Eye Tracker file (*.csv) that was obtained from hyperscanning2-redesign experiment \n
    Arguments :
        - path_2_csv_files (str) : path to raw Eye tracker file \n
        - path_2_save_experimental_file (str) : path to save extracted experimental file for each condition \n
        - labelsequence (int) : order of pre-defined label sequence, 1 (averted) is default \n
        - bad_files (list) (optional) : file name of raw EyeTracker file, e.g., EyeTracker-S8.csv, that wants to be skipped to process
    Return :
        Extracted *.csv file for each condition of hand (finger pointing and tracking).
        There are 6 files in total for each participant.


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

    # Iterate all file names

    for i in tqdm(range(len(list_file_names)), desc="In progress"):

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
                fileName = list_file_names[i]

                # Change to a folder where original CSV files are stored
                os.chdir(path_2_csv_files)

                print("Processing file : " + list_file_names[i])

                # Read each file by using pandas
                df = pd.read_csv(fileName, delimiter=",")

                df["UnixTimeStamp"] = df.UnixTimeStamp.apply(
                    lambda x: "9999999" if "BEGIN" in x else x
                )
                df["UnixTimeStamp"] = df.UnixTimeStamp.apply(
                    lambda x: "9999999" if "END" in x else x
                )

                # Turn the UnixTimeStamp column into a list (we need the marker later on)
                markers = df["UnixTimeStamp"].tolist()

                # Convert string value to integer number
                markers = list(map(int, markers))

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
                chosenOrder = listOfOrders[labelsequence - 1]

                # Get the experimental markers' indices and extract the data
                indicesofExperimentalMarkers = indicesOfMarkers[12:]

                # Chunk the data based on opening and closing markers
                chunkedData = []
                for i in range(0, len(indicesofExperimentalMarkers), 2):

                    chunkedData.append(
                        df.iloc[
                            indicesofExperimentalMarkers[
                                i
                            ] : indicesofExperimentalMarkers[i + 1],
                            :,
                        ]
                    )

                # Match pattern EyeTracker-Sx (x = any number)
                regex = r"\D{10}-S\d+"

                # Create filename that will be used for each condition. There are 6 conditions. See oddOrder1 or evenOrder1
                extracted_file_name_4_experimental = []
                for i in chosenOrder:
                    extracted_file_name = re.search(regex, fileName)
                    extracted_file_name_4_experimental.append(
                        fileName[
                            extracted_file_name.start() : extracted_file_name.end()
                        ]
                        + "-"
                        + i
                        + "_raw.csv"
                    )

                # Save the chunkedData into a separate csv file
                for i, val in tqdm(enumerate(chunkedData), desc="Saving process..."):

                    # Convert array into dataframe
                    df_chunkedData = pd.DataFrame(val)

                    # Save dataframe into csv
                    os.chdir(path_2_save_experimental_file)
                    df_chunkedData.to_csv(
                        extracted_file_name_4_experimental[i], sep=(",")
                    )

    print(
        f"All experimental files of eye data have been saved in csv format in this path {path_2_save_experimental_file}"
    )
