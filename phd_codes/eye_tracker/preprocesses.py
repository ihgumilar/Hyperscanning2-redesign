## Relevant packages
import chunk
import numbers
import os
import pickle
import re
import warnings
from os import listdir
from typing import List

import mne
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import signal
from tqdm import tqdm


class Extract_Eye_Exp2_Redesign:
    def extract_baseline_eye_data(
        self,
        path_2_csv_files: str,
        path_2_save_baseline_file: str,
        labelsequence: int = 1,
        bad_files: list = [],
    ):

        """
        Objective: Extract baseline data from raw Eye Tracker file (*.csv) that was obtained from hyperscanning2-redesign experiment \n

        Parameters :
                    - path_2_csv_files (str) : path to raw Eye tracker file \n
                    - path_2_save_baseline_file (str) : path to save extracted baseline file for each condition \n
                    - bad_files (list) (optional) : file name of raw EyeTracker file, e.g., EyeTracker-S8.csv, that wants to be skipped to process
                    - labelsequence (int) : order of pre-defined label sequence, 1 (averted) is default \n

        Output :
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
                                    indicesofBaselineMarkers[
                                        i
                                    ] : indicesofBaselineMarkers[i + 1],
                                    :,
                                ]
                            )

                        # Match pattern EyeTracker-Sx (x = any number)
                        regex = r"\D{10}-S\d+"

                        # Create filename that will be used for each condition. There are 6 conditions. See oddOrder1 or evenOrder1
                        extracted_file_name_4_baseline = []
                        for i in chosenOrder:
                            # The re.search() method returns a Match object ( i.e., re.Match).
                            # This match object contains the following two items.
                            # 1. The tuple object contains the start and end index of a successful match.
                            # 2. Second, it contains an actual matching value that we can retrieve using a group() method.

                            extracted_file_name = re.search(regex, fileName)

                            # The tuple object contains the start and end index of a successful match.
                            # So, ignore the error as being underlined there
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

    # %% [markdown]
    # ## Extract_experiment_eye_data function

    # %% Extract experimental data

    def extract_experiment_eye_data(
        self,
        path_2_csv_files: str,
        path_2_save_experimental_file: str,
        labelsequence_experiment: list,
        bad_files: list = [],
    ):

        """
        Objective  : Extract experimental data from raw Eye Tracker file (*.csv) that was obtained from hyperscanning2-redesign experiment \n

        Parameters :
                    - path_2_csv_files (str) : path to raw Eye tracker file \n
                    - path_2_save_experimental_file (str) : path to save extracted experimental file for each condition \n
                    - labelsequence_experiment (list) : order of pre-defined label sequence \n
                    - bad_files (list) (optional) : file name of raw EyeTracker file, e.g., EyeTracker-S8.csv, that wants to be skipped to process

        Output :
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
                        # The re.search() method returns a Match object ( i.e., re.Match).
                        # This match object contains the following two items.
                        # 1. The tuple object contains the start and end index of a successful match.
                        # 2. Second, it contains an actual matching value that we can retrieve using a group() method.

                        extracted_file_name = re.search(regex, fileName)
                        # The tuple object contains the start and end index of a successful match.
                        # So, ignore the error as being underlined there

                        extracted_file_name_4_experimental.append(
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
                        os.chdir(path_2_save_experimental_file)
                        df_chunkedData.to_csv(
                            extracted_file_name_4_experimental[i], sep=(",")
                        )

        print(
            f"All experimental files of eye data have been saved in csv format in this path {path_2_save_experimental_file}"
        )


class Combine_Eye_Exp2_Redesign:
    def __init__(self, n_raw_files: int):
        """
        n_raw_files(int) = Number of raw files. Must be EVEN Number
                           because it is always in pair
        """
        self.number_raw_files = n_raw_files

    def combine_pre_averted_baseline(
        self, raw_dir_baseline: str, raw_combined_baseline_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of PRE-training during baseline stage.

        Parameters : - raw_dir_baseline(str) : path to directory that contains separate eye tracker data files of baseline
                     - raw_combined_baseline_data_directory(str) : path to store combined baseline-data of eye tracker

        Output     : Combined eye tracker data in format .csv
        """

        # Change to directory which stores raw baseline data (not combined)
        os.chdir(raw_dir_baseline)
        for i in tqdm(range(self.number_raw_files), desc="Combining pre averted..."):  # type: ignore

            # Pre-averted (for processing subject 1 - 9)
            if i < 9:

                # Load averted pre right
                averted_pre_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_pre_right_point_raw.csv"
                )
                averted_pre_right_odd_subject = pd.read_csv(
                    averted_pre_right_odd_subject_file_name
                )

                # Load Load averted pre left
                averted_pre_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_pre_left_point_raw.csv"
                )
                averted_pre_left_odd_subject = pd.read_csv(
                    averted_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

            # Pre-averted (for processing subject 10 onwards)
            else:
                # Load averted pre right
                averted_pre_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_pre_right_point_raw.csv"
                )
                averted_pre_right_odd_subject = pd.read_csv(
                    averted_pre_right_odd_subject_file_name
                )

                # Load Load averted pre left
                averted_pre_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_pre_left_point_raw.csv"
                )
                averted_pre_left_odd_subject = pd.read_csv(
                    averted_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

    def combine_post_averted_baseline(
        self, raw_dir_baseline: str, raw_combined_baseline_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during baseline stage.

        Parameters : - raw_dir_baseline(str) : path to directory that contains separate eye tracker data files of baseline
                     - raw_combined_baseline_data_directory(str) : path to store combined baseline-data of eye tracker

        Output     : Combined eye tracker data in format .csv
        """
        # Change to directory which stores raw baseline data (not combined)
        os.chdir(raw_dir_baseline)

        for i in tqdm(range(self.number_raw_files), desc="Combining post averted..."):  # type: ignore

            # Pre-averted (for processing subject 1 - 9)
            if i < 9:

                # Load averted post right
                averted_post_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_post_right_point_raw.csv"
                )
                averted_post_right_odd_subject = pd.read_csv(
                    averted_post_right_odd_subject_file_name
                )

                # Load Load averted post left
                averted_post_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_post_left_point_raw.csv"
                )
                averted_post_left_odd_subject = pd.read_csv(
                    averted_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_right_odd_subject, averted_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_left_odd_subject, averted_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

            # Pre-averted (for processing subject 10 onwards)
            else:
                # Load averted post right
                averted_post_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_post_right_point_raw.csv"
                )
                averted_post_right_odd_subject = pd.read_csv(
                    averted_post_right_odd_subject_file_name
                )

                # Load Load averted post left
                averted_post_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_post_left_point_raw.csv"
                )
                averted_post_left_odd_subject = pd.read_csv(
                    averted_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_right_odd_subject, averted_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_left_odd_subject, averted_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_baseline_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

    def combine_pre_averted_experiment(
        self, raw_dir_experiment: str, raw_combined_experimental_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during experimental stage.

        Parameters : - raw_dir_experiment(str) : path to directory that contains separate eye tracker data files of experiment
                     - raw_combined_experimental_data_directory(str) : path to store combined-experimental data of eye tracker

        Output     : Combined eye tracker data in format .csv

        """

        # Change to directory which stores raw experimental data (not combined)
        os.chdir(raw_dir_experiment)

        for i in tqdm(range(self.number_raw_files), desc="Combining pre averted..."):  # type: ignore

            # Pre-averted (for processing subject 1 - 9)
            if i < 9:

                # Load averted pre right
                averted_pre_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_pre_right_point_raw.csv"
                )
                averted_pre_right_odd_subject = pd.read_csv(
                    averted_pre_right_odd_subject_file_name
                )

                # Load Load averted pre left
                averted_pre_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_pre_left_point_raw.csv"
                )
                averted_pre_left_odd_subject = pd.read_csv(
                    averted_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

            # Pre-averted (for processing subject 10 onwards)
            else:
                # Load averted pre right
                averted_pre_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_pre_right_point_raw.csv"
                )
                averted_pre_right_odd_subject = pd.read_csv(
                    averted_pre_right_odd_subject_file_name
                )

                # Load Load averted pre left
                averted_pre_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_pre_left_point_raw.csv"
                )
                averted_pre_left_odd_subject = pd.read_csv(
                    averted_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_averted_files = pd.concat(
                        [averted_pre_left_odd_subject, averted_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-averted baseline
                    combined_pre_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-averted baseline file to csv
                    combined_pre_averted_files.to_csv(combined_pre_averted_files_label)

    def combine_post_averted_experiment(
        self, raw_dir_experiment: str, raw_combined_experimental_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during experimental stage.

        Parameters : - raw_dir_experiment(str) : path to directory that contains separate eye tracker data files of experiment
                     - raw_combined_experimental_data_directory(str) : path to store combined-experimental data of eye tracker

        Output     : Combined eye tracker data in format .csv

        """

        # Change to directory which stores raw experimental data (not combined)
        os.chdir(raw_dir_experiment)

        for i in tqdm(range(self.number_raw_files), desc="Combining post averted..."):  # type: ignore

            # post-averted (for processing subject 1 - 9)
            if i < 9:

                # Load averted post right
                averted_post_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_post_right_point_raw.csv"
                )
                averted_post_right_odd_subject = pd.read_csv(
                    averted_post_right_odd_subject_file_name
                )

                # Load Load averted post left
                averted_post_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-averted_post_left_point_raw.csv"
                )
                averted_post_left_odd_subject = pd.read_csv(
                    averted_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_right_odd_subject, averted_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_left_odd_subject, averted_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-averted_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

            # post-averted (for processing subject 10 onwards)
            else:
                # Load averted post right
                averted_post_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_post_right_point_raw.csv"
                )
                averted_post_right_odd_subject = pd.read_csv(
                    averted_post_right_odd_subject_file_name
                )

                # Load Load averted post left
                averted_post_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-averted_post_left_point_raw.csv"
                )
                averted_post_left_odd_subject = pd.read_csv(
                    averted_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_right_odd_subject, averted_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_averted_files = pd.concat(
                        [averted_post_left_odd_subject, averted_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-averted baseline
                    combined_post_averted_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-averted_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-averted baseline file to csv
                    combined_post_averted_files.to_csv(
                        combined_post_averted_files_label
                    )

    def combine_pre_direct_experiment(
        self, raw_dir_experiment: str, raw_combined_experimental_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during experimental stage.

        Parameters : - raw_dir_experiment(str) : path to directory that contains separate eye tracker data files of experiment
                     - raw_combined_experimental_data_directory(str) : path to store combined-experimental data of eye tracker

        Output     : Combined eye tracker data in format .csv

        """

        # Change to directory which stores raw experimental data (not combined)
        os.chdir(raw_dir_experiment)

        for i in tqdm(range(self.number_raw_files), desc="Combining pre direct..."):  # type: ignore

            # Pre-direct (for processing subject 1 - 9)
            if i < 9:

                # Load direct pre right
                direct_pre_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-direct_pre_right_point_raw.csv"
                )
                direct_pre_right_odd_subject = pd.read_csv(
                    direct_pre_right_odd_subject_file_name
                )

                # Load Load direct pre left
                direct_pre_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-direct_pre_left_point_raw.csv"
                )
                direct_pre_left_odd_subject = pd.read_csv(
                    direct_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_direct_files = pd.concat(
                        [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-direct baseline
                    combined_pre_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-direct_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-direct baseline file to csv
                    combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_direct_files = pd.concat(
                        [direct_pre_left_odd_subject, direct_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-direct baseline
                    combined_pre_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-direct_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-direct baseline file to csv
                    combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

            # Pre-direct (for processing subject 10 onwards)
            else:
                # Load direct pre right
                direct_pre_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-direct_pre_right_point_raw.csv"
                )
                direct_pre_right_odd_subject = pd.read_csv(
                    direct_pre_right_odd_subject_file_name
                )

                # Load Load direct pre left
                direct_pre_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-direct_pre_left_point_raw.csv"
                )
                direct_pre_left_odd_subject = pd.read_csv(
                    direct_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_direct_files = pd.concat(
                        [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-direct baseline
                    combined_pre_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-direct_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-direct baseline file to csv
                    combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_direct_files = pd.concat(
                        [direct_pre_left_odd_subject, direct_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-direct baseline
                    combined_pre_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-direct_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-direct baseline file to csv
                    combined_pre_direct_files.to_csv(combined_pre_direct_files_label)

    def combine_post_direct_experiment(
        self, raw_dir_experiment: str, raw_combined_experimental_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during experimental stage.

        Parameters : - raw_dir_experiment(str) : path to directory that contains separate eye tracker data files of experiment
                     - raw_combined_experimental_data_directory(str) : path to store combined-experimental data of eye tracker

        Output     : Combined eye tracker data in format .csv

        """

        # Change to directory which stores raw experimental data (not combined)
        os.chdir(raw_dir_experiment)

        for i in tqdm(range(self.number_raw_files), desc="Combining post direct..."):  # type: ignore

            # post-direct (for processing subject 1 - 9)
            if i < 9:

                # Load direct post right
                direct_post_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-direct_post_right_point_raw.csv"
                )
                direct_post_right_odd_subject = pd.read_csv(
                    direct_post_right_odd_subject_file_name
                )

                # Load Load direct post left
                direct_post_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-direct_post_left_point_raw.csv"
                )
                direct_post_left_odd_subject = pd.read_csv(
                    direct_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_direct_files = pd.concat(
                        [direct_post_right_odd_subject, direct_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-direct baseline
                    combined_post_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-direct_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-direct baseline file to csv
                    combined_post_direct_files.to_csv(combined_post_direct_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_direct_files = pd.concat(
                        [direct_post_left_odd_subject, direct_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-direct baseline
                    combined_post_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-direct_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-direct baseline file to csv
                    combined_post_direct_files.to_csv(combined_post_direct_files_label)

            # post-direct (for processing subject 10 onwards)
            else:
                # Load direct post right
                direct_post_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-direct_post_right_point_raw.csv"
                )
                direct_post_right_odd_subject = pd.read_csv(
                    direct_post_right_odd_subject_file_name
                )

                # Load Load direct post left
                direct_post_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-direct_post_left_point_raw.csv"
                )
                direct_post_left_odd_subject = pd.read_csv(
                    direct_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_direct_files = pd.concat(
                        [direct_post_right_odd_subject, direct_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-direct baseline
                    combined_post_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-direct_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-direct baseline file to csv
                    combined_post_direct_files.to_csv(combined_post_direct_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_direct_files = pd.concat(
                        [direct_post_left_odd_subject, direct_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-direct baseline
                    combined_post_direct_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-direct_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-direct baseline file to csv
                    combined_post_direct_files.to_csv(combined_post_direct_files_label)

    def combine_pre_natural_experiment(
        self, raw_dir_experiment: str, raw_combined_experimental_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during experimental stage.

        Parameters : - raw_dir_experiment(str) : path to directory that contains separate eye tracker data files of experiment
                     - raw_combined_experimental_data_directory(str) : path to store combined-experimental data of eye tracker

        Output     : Combined eye tracker data in format .csv

        """

        # Change to directory which stores raw experimental data (not combined)
        os.chdir(raw_dir_experiment)

        for i in tqdm(range(self.number_raw_files), desc="Combining pre natural..."):  # type: ignore

            # Pre-natural (for processing subject 1 - 9)
            if i < 9:

                # Load natural pre right
                natural_pre_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-natural_pre_right_point_raw.csv"
                )
                natural_pre_right_odd_subject = pd.read_csv(
                    natural_pre_right_odd_subject_file_name
                )

                # Load Load natural pre left
                natural_pre_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-natural_pre_left_point_raw.csv"
                )
                natural_pre_left_odd_subject = pd.read_csv(
                    natural_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_natural_files = pd.concat(
                        [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-natural baseline
                    combined_pre_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-natural_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-natural baseline file to csv
                    combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_natural_files = pd.concat(
                        [natural_pre_left_odd_subject, natural_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-natural baseline
                    combined_pre_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-natural_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-natural baseline file to csv
                    combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

            # Pre-natural (for processing subject 10 onwards)
            else:
                # Load natural pre right
                natural_pre_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-natural_pre_right_point_raw.csv"
                )
                natural_pre_right_odd_subject = pd.read_csv(
                    natural_pre_right_odd_subject_file_name
                )

                # Load Load natural pre left
                natural_pre_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-natural_pre_left_point_raw.csv"
                )
                natural_pre_left_odd_subject = pd.read_csv(
                    natural_pre_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_pre_natural_files = pd.concat(
                        [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
                    )

                    # Create  file name for combine files of pre-natural baseline
                    combined_pre_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-natural_pre_right_left_point_combined_raw.csv"
                    )

                    # Save combine pre-natural baseline file to csv
                    combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_pre_natural_files = pd.concat(
                        [natural_pre_left_odd_subject, natural_pre_right_odd_subject]
                    )

                    # Create  file name for combine files of pre-natural baseline
                    combined_pre_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-natural_pre_left_right_point_combined_raw.csv"
                    )

                    # Save combine pre-natural baseline file to csv
                    combined_pre_natural_files.to_csv(combined_pre_natural_files_label)

    def combine_post_natural_experiment(
        self, raw_dir_experiment: str, raw_combined_experimental_data_directory: str
    ):

        """
         Objective :
                    After the data of eye tracker has been extracted, it is separated between left and right hand data \n
                    Due to that, we need to combine both data by using this function. During pre-training, participants need to point \n
                    their hands with right and left alternatively. It is 1 minute for each hand. Since there are \n
                    two hands, then it was 2 minutes for both hands. This function is to combine data of eye tracker \n
                    of POST-training during experimental stage.

        Parameters : - raw_dir_experiment(str) : path to directory that contains separate eye tracker data files of experiment
                     - raw_combined_experimental_data_directory(str) : path to store combined-experimental data of eye tracker

        Output     : Combined eye tracker data in format .csv

        """

        # Change to directory which stores raw experimental data (not combined)
        os.chdir(raw_dir_experiment)

        for i in tqdm(range(self.number_raw_files), desc="Combining post natural..."):  # type: ignore

            # post-natural (for processing subject 1 - 9)
            if i < 9:

                # Load natural post right
                natural_post_right_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-natural_post_right_point_raw.csv"
                )
                natural_post_right_odd_subject = pd.read_csv(
                    natural_post_right_odd_subject_file_name
                )

                # Load Load natural post left
                natural_post_left_odd_subject_file_name = (
                    "EyeTracker-S0" + str(i + 1) + "-natural_post_left_point_raw.csv"
                )
                natural_post_left_odd_subject = pd.read_csv(
                    natural_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_natural_files = pd.concat(
                        [natural_post_right_odd_subject, natural_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-natural baseline
                    combined_post_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-natural_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-natural baseline file to csv
                    combined_post_natural_files.to_csv(
                        combined_post_natural_files_label
                    )

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_natural_files = pd.concat(
                        [natural_post_left_odd_subject, natural_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-natural baseline
                    combined_post_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S0"
                        + str(i + 1)
                        + "-natural_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-natural baseline file to csv
                    combined_post_natural_files.to_csv(
                        combined_post_natural_files_label
                    )

            # post-natural (for processing subject 10 onwards)
            else:
                # Load natural post right
                natural_post_right_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-natural_post_right_point_raw.csv"
                )
                natural_post_right_odd_subject = pd.read_csv(
                    natural_post_right_odd_subject_file_name
                )

                # Load Load natural post left
                natural_post_left_odd_subject_file_name = (
                    "EyeTracker-S" + str(i + 1) + "-natural_post_left_point_raw.csv"
                )
                natural_post_left_odd_subject = pd.read_csv(
                    natural_post_left_odd_subject_file_name
                )

                # Check if i (index in this looping) == EVEN number that takes ODD actual subject no, then labeling the file name of hand is RIGHT-LEFT
                # For example, i = 0 actually takes S01 and keeps going...
                if (i % 2) == 0:

                    # Combine RIGHT => LEFT hand data
                    combined_post_natural_files = pd.concat(
                        [natural_post_right_odd_subject, natural_post_left_odd_subject]
                    )

                    # Create  file name for combine files of post-natural baseline
                    combined_post_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-natural_post_right_left_point_combined_raw.csv"
                    )

                    # Save combine post-natural baseline file to csv
                    combined_post_natural_files.to_csv(
                        combined_post_natural_files_label
                    )

                # Check if i (index in this looping) == ODD number that takes EVEN actual subject no, then labeling the file name of hand is LEFT_RIGHT
                # For example, i = 1 actually takes S02 and keeps going...
                else:

                    # Combine LEFT => RIGHT hand data
                    combined_post_natural_files = pd.concat(
                        [natural_post_left_odd_subject, natural_post_right_odd_subject]
                    )

                    # Create  file name for combine files of post-natural baseline
                    combined_post_natural_files_label = (
                        raw_combined_experimental_data_directory
                        + "EyeTracker-S"
                        + str(i + 1)
                        + "-natural_post_left_right_point_combined_raw.csv"
                    )

                    # Save combine post-natural baseline file to csv
                    combined_post_natural_files.to_csv(
                        combined_post_natural_files_label
                    )


class Clean_Eye_Exp2_Redesign:
    def delete_epoch_eye_tracker(
        self,
        eyetracker_data_path: str,
        path2save_cleaned_data: str,
        file_tag: str,
        indices: List[List[int]],
    ):

        """
        Objective : Since EEG data has been cleaned (delete bad data) then the same part of eye tracker data will also
                    need to be deleted. This function will do the job

        Parameters :
                    - eyetracker_data_path (str)  : path to where extracted eye data has been stored
                    - path2save_cleaned_data(str) : path to save cleaned eye data
                    - file_tag(str)               : averted_pre, averted_post, direct_pre, direct_post
                    - indices (List[List[int]])   : List of indices of deleted epochs of EEG data
                                                    Note : Load pkl file which consists of deleted EEG indices

                                                   with open("list_deleted_epoch_indices_averted_pre.pkl", "rb") as handle:
                                                        deleted_epochs_indices_averted_pre = pickle.load(handle)

                                                    Take the variable of deleted_epochs_indices_averted_pre as input


        Outputs    : Cleaned csv files where the same part of EEG data has been deleted
        """

        # List specified files based on file_tag parameter that has been defined
        files = [file for file in os.listdir(eyetracker_data_path) if file_tag in file]
        files = sorted(files, key=lambda x: int(x.partition("S")[2].partition("-")[0]))
        begin = 0
        end = len(files)
        step = 2
        counter = 0

        for idx in tqdm(
            range(begin, end, step),
            desc="Have some coffee bro, while we are processing your files :) ",
        ):

            # df1 = pd.read_csv(eyetracker_data_path + files[0])
            df1 = pd.read_csv(eyetracker_data_path + files[idx])
            df2 = pd.read_csv(eyetracker_data_path + files[idx + 1])

            # Drop missing values
            df1 = df1.dropna()
            df2 = df2.dropna()
            dfNew1 = pd.DataFrame()
            dfNew2 = pd.DataFrame()

            # Resample the data (125 sampling rate x 120 seconds = 15000 rows)
            for column in df1.columns:
                if is_numeric_dtype(df1[column]):
                    dfNew1[column] = signal.resample(df1[column].tolist(), 15000)

            for column in df2.columns:
                if is_numeric_dtype(df2[column]):
                    dfNew2[column] = signal.resample(df2[column].tolist(), 15000)

            # Create dictionary which contains indices to delete
            key_counter = 0
            labels_indices = {}
            for val in range(0, 15000, 125):
                idx_start = val
                idx_end = val + 125
                labels_indices.update({key_counter: [idx_start, idx_end]})
                key_counter += 1

            if idx >= 1:
                counter += 1

            labels_indices_2_delete = []

            for individual_indices in indices[counter]:
                labels_indices_2_delete.append(labels_indices[individual_indices])

            # Mark with 1 for rows to be deleted
            for val in labels_indices_2_delete:
                begin_idx_delete = val[0]
                end_idx_delete = val[1]

                dfNew1.loc[
                    dfNew1.index[begin_idx_delete:end_idx_delete], "Mark2Delete"
                ] = 1
                dfNew2.loc[
                    dfNew2.index[begin_idx_delete:end_idx_delete], "Mark2Delete"
                ] = 1

                # Delete rows which have marked 1 in column Mark2Delete
                dfNew1 = dfNew1[dfNew1["Mark2Delete"] != 1]
                dfNew2 = dfNew2[dfNew2["Mark2Delete"] != 1]

                # Drop column of Mark2Delete
                dfNew1.drop("Mark2Delete", axis=1, inplace=True)
                dfNew2.drop("Mark2Delete", axis=1, inplace=True)

            # Save cleaned data
            # path2save_cleaned_data = '/hpc/igum002/codes/frontiers_hyperscanning2/eye_tracker_data_clean_new/'
            df1_name = files[idx]
            df2_name = files[idx + 1]
            dfNew1.to_csv(
                path2save_cleaned_data + df1_name[: df1_name.rfind(".")] + "_clean.csv"
            )
            dfNew2.to_csv(
                path2save_cleaned_data + df2_name[: df2_name.rfind(".")] + "_clean.csv"
            )
            print("Processed : " + df1_name)
            print("Processed : " + df2_name)

        print("The eye tracker files have been cleaned and you are good to go, Bro !")
