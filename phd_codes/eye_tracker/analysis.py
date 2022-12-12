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

# %% [markdown]
# ## Relevant packages

import glob
import os
import re
import sys
from math import atan, degrees
from typing import List

import matplotlib.cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_bar
from EEG.stats import Connections
from eye_tracker.analysis import EyeAnalysis

# add alpha (transparency) to a colormap (with background picture)
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame

# %% [markdown]
# ## Low level module / class Degree

# %%
class Degree:
    """
    Low-level module
    """

    ########################## Functions to convert to degree and check within fovea or not #################

    # Formula to convert cartesian to degree
    def gaze_direction_in_x_axis_degree(self, x: float or int, y: float or int):
        """
        Objective  : Convert cartesian value to degree for gaze direction in X axis
                     Eye tracker data that is produced by HTC Vive pro is in cartesian

                     See "Right hand rule coordinate" for further information

        Parameters : - x (float or int) :  data in X axis
                     - y (float or int) :  data in Y axis

        Output      : - degree (float or int) : if it is float, then it has been rounded to 2 decimals

        """
        try:
            degree = degrees(atan(y / x))  # Opposite / adjacent
        except ZeroDivisionError:
            degree = 0.0
        return round(degree, 2)

    def gaze_direction_in_y_axis_degree(self, y: float or int, z: float or int):

        """
        Objective  : Convert cartesian value to degree for gaze direction in Y axis
                     Eye tracker data that is produced by HTC Vive pro is in cartesian

                     See "Right hand rule coordinate" for further information

        Parameters : - y (float or int) :  data in Y axis
                     - z (float or int) :  data in Z axis

        Output      : - degree (float or int) : if it is float, then it has been rounded to 2 decimals

        """

        try:
            degree = degrees(atan(z / y))  # Opposite / adjacent
        except ZeroDivisionError:
            degree = 0.0
        return round(degree, 2)

    def check_degree_within_fovea(self, gaze_direction: float or int):

        """
        Objective   : We need to check if the degree is within fovea or not, which indicates something is being looked.
                      In total 30 degrees where human can recognize an object. So we need to divide by 2 \n
                      Half right and half left

        Parameters  : gaze direction (float or int) : gaze direction which is in degree
                      If we refer to pandas dataframe, it is related to either one of these columns

                      - GazeDirectionRight(X)Degree
                      - GazeDirectionRight(Y)Degree

                      - GazeDirectionLeft(X)Degree
                      - GazeDirectionLeft(Y)Degree

        Output (int) :  1 = within the area of fovea (30 degrees) and
                        0 = NOT within area of fovea

        """

        if (gaze_direction <= 15) & (gaze_direction >= 0):
            return 1
        elif (gaze_direction >= -15) & (gaze_direction <= 0):
            return 1
        else:
            return 0


# %% [markdown]
# ## Low level module / class Looking

# %%
class Looking:
    """
    Low-level module
    """

    def looking_percentage(
        self,
        odd_dataframe: DataFrame,
        even_dataframe: DataFrame,
        srate: int = 125,
        threshold: int = 13,
    ):

        """
        Objective  : Count how many times each pair "really" looks at each other throughout the experiment
                    This will look at per second.  "Really" is determined by a previous research which indicates
                    that humans are conscious look at each other within 100ms,
                    which is equal to there are 13 times of "looking" value within column of FoveaOdd or FoveaEven

        Parameters : - odd_dataframe (pandas dataframe) :  dataframe of odd participant
                    - even_dataframe (pandas dataframe) :  dataframe of even participant
                    - srate (int) : which indicates the step / per second that we will check whether
                                    the pair looking or not
                    - threshold (int) : threshold to determine whether the pair "really" looks or not
                                        if there are at least 13 times of "looking" within a second (srate)
                                        under the column of FoveaOdd or FoveaEven, then it is considered "really" looking
                        Note : Kindly refer to this research to see the threshold of 30 (100ms)
                        https://journals.sagepub.com/doi/abs/10.1111/j.1467-9280.2006.01750.x?casa_token=AYU81Dg2DAMAAAAA%3Asy9nVGA6NjQPFuRthQW5eCZl9V06TpqV2OgtYbUFPwVKCV4so2PlVJrBWo01EfiSX-yNHul7mX_DlYk&journalCode=pssa

        Output      : - percent_look (float) : percentage of looking of a pair throughout the experiment

        """

        list_look = []
        # To chunk the data for every n raws (which indicate per second). Refer to srate
        for i in range(0, odd_dataframe.shape[0], srate):
            count = 0
            # To loop the series or data in a dataframe so that we can check if they are matched or not
            for j in range(i, i + srate):
                if (
                    odd_dataframe.iloc[j]["FoveaOdd"] == "look"
                    and even_dataframe.iloc[j]["FoveaEven"] == "look"
                ):
                    count += 1

            # each element of the list represents whether the pair looks at each other or not within a second
            list_look += [1 if count >= threshold else 0]

        # Percentage of looking
        percent_look = sum(list_look) / len(list_look) * 100
        # Get the last two digits only
        percent_look = float("{:.2f}".format(percent_look))

        return percent_look


# %% [markdown]
# ## Abstract module / class

# %%
class _IntermediaryEye:
    """
    An abstract module
    """

    def __init__(
        self,
    ):
        self.degree = Degree()
        self.looking = Looking()

    def convert_gaze_direction_in_x_axis_to_degree(
        self, x: float or int, y: float or int
    ):
        """
        Objective  : Convert cartesian value to degree for gaze direction in X axis
                     Eye tracker data that is produced by HTC Vive pro is in cartesian

                     See "Right hand rule coordinate" for further information

        Parameters : - x (float or int) :  data in X axis
                     - y (float or int) :  data in Y axis

        Output      : - degree (float or int) : if it is float, then it has been rounded to 2 decimals

        """
        degree = self.degree.gaze_direction_in_x_axis_degree(x, y)
        return degree

    def convert_gaze_direction_in_y_axis_to_degree(
        self, y: float or int, z: float or int
    ):
        """
        Objective  : Convert cartesian value to degree for gaze direction in Y axis
                     Eye tracker data that is produced by HTC Vive pro is in cartesian

                     See "Right hand rule coordinate" for further information

        Parameters : - y (float or int) :  data in Y axis
                     - z (float or int) :  data in Z axis

        Output      : - degree (float or int) : if it is float, then it has been rounded to 2 decimals

        """
        degree = self.degree.gaze_direction_in_y_axis_degree(y, z)
        return degree

    def check_degree_within_fovea(self, gaze_direction: float or int):
        """
        Objective   : We need to check if the degree is within fovea or not, which indicates something is being looked.
                      In total 30 degrees where human can recognize an object. So we need to divide by 2 \n
                      half right and half left

        Parameters  : gaze direction (float or int) : gaze direction which is in degree \n
                      If we refer to pandas dataframe, it is related to either one of these columns

                        - GazeDirectionRight(X)Degree
                        - GazeDirectionRight(Y)Degree

                        - GazeDirectionLeft(X)Degree
                        - GazeDirectionLeft(Y)Degree

        Output (int) :  1 = within the area of fovea (30 degrees) and
                        0 = NOT within area of fovea

        """
        within_or_not_fovea = self.degree.check_degree_within_fovea(gaze_direction)

        return within_or_not_fovea

    def looking_percentage(
        self,
        odd_dataframe: DataFrame,
        even_dataframe: DataFrame,
        srate: int = 125,
        threshold: int = 13,
    ):
        """
        Objective  : Count how many times each pair "really" looks at each other throughout the experiment
                    This will look at per second.  "Really" is determined by a previous research which indicates
                    that humans are conscious look at each other within 100ms,
                    which is equal to there are 13 times of "looking" value within column of FoveaOdd or FoveaEven

        Parameters : - odd_dataframe (pandas dataframe) :  dataframe of odd participant
                    - even_dataframe (pandas dataframe) :  dataframe of even participant
                    - srate (int) : which indicates the step / per second that we will check whether
                                    the pair looking or not
                    - threshold (int) : threshold to determine whether the pair "really" looks or not
                                        if there are at least 13 times of "looking" within a second (srate)
                                        under the column of FoveaOdd or FoveaEven, then it is considered "really" looking
                        Note : Kindly refer to this research to see the threshold of 30 (100ms)
                        https://journals.sagepub.com/doi/abs/10.1111/j.1467-9280.2006.01750.x?casa_token=AYU81Dg2DAMAAAAA%3Asy9nVGA6NjQPFuRthQW5eCZl9V06TpqV2OgtYbUFPwVKCV4so2PlVJrBWo01EfiSX-yNHul7mX_DlYk&journalCode=pssa

        Output      : - percent_look (float) : percentage of looking of a pair throughout the experiment

        """
        self.srate = 125
        self.threshold = 13
        percent_look = self.looking.looking_percentage(
            odd_dataframe, even_dataframe, self.srate, self.threshold
        )
        return percent_look


# %% [markdown]
# ## High-level module class of EyeAnalysis

# %%
class EyeAnalysis:

    """
    High-level module
    """

    def __init__(self):
        self.__intermediaryEye = _IntermediaryEye()

    def eye_data_analysis(self, path2files: str, tag: str):

        """
        Objective  : Analyze all cleaned eye tracker data, eg. averted_pre.
                     It also involves pre-processing (replacing missing values with average value of columns
                     where they are). It calculates how much each pair looks at each other \n
                     throughout the experiment(each eye condition)

        Parameters : - path2files (str): Path to a directory where all cleaned eye tracker files are stored
                     - tag (str): eye gaze condition, ie. averted_pre, averted_post, direct_pre, direct_post, natural_pre, natural_post

        Output:      - looking_percentage_all_pairs (list) : Each element represents the percentage  \n
                                                            of looking of each pair throughout the experiment

        """

        gaze_keyword = "/*" + tag + "*.csv"
        pre_files = glob.glob(path2files + gaze_keyword)
        pattern = re.compile(r"[S]+(\d+)\-")
        files_pre_odd = []
        files_pre_even = []
        looking_percentage_all_pairs = []

        for idx, file in enumerate(pre_files):

            # Put into a list for ODD subjects - Refer to filename
            if (idx % 2) == 0:
                files_pre_odd.append(file)

            # Put into different list for EVEN subjects - Refer to filename
            else:
                files_pre_even.append(file)

        ############################################### Odd subject ###############################################
        # Combine all pre odd files

        with alive_bar(
            len(files_pre_odd), title="Eye Data(" + tag + ")", force_tty=True
        ) as bar:

            for idx, filename in enumerate(files_pre_odd):

                df_odd = pd.read_csv(filename, index_col=None, header=0)

                # Replace missing values with averages of columns where they are
                df_odd.fillna(df_odd.mean(), inplace=True)

                # Remove space before column names
                df_odd_new_columns = df_odd.columns.str.replace(" ", "")
                df_odd.columns = df_odd_new_columns

                # convert cartesian to degree of eye data

                # Gaze direction (right eye)
                df_odd["GazeDirectionRight(X)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionRight(X)"], x["GazeDirectionRight(Y)"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionRight(Y)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionRight(Y)"], x["GazeDirectionRight(Z)"]
                    ),
                    axis=1,
                )

                # Gaze direction (left eye)
                df_odd["GazeDirectionLeft(X)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionLeft(X)"], x["GazeDirectionLeft(Y)"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionLeft(Y)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionLeft(Y)"], x["GazeDirectionLeft(Z)"]
                    ),
                    axis=1,
                )

                # check degree_within_fovea or not
                df_odd["GazeDirectionRight(X)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(X)Degree"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionRight(Y)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(Y)Degree"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionLeft(X)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(X)Degree"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionLeft(Y)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(Y)Degree"]
                    ),
                    axis=1,
                )

                # Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (odd subjects)
                df_odd["FoveaOdd"] = (
                    df_odd["GazeDirectionRight(X)inFovea"]
                    + df_odd["GazeDirectionLeft(X)inFovea"]
                    + df_odd["GazeDirectionRight(Y)inFovea"]
                    + df_odd["GazeDirectionLeft(Y)inFovea"]
                )
                df_odd.loc[df_odd["FoveaOdd"] > 1, "FoveaOdd"] = 1

                # Change 1 => look , 0 => not look (odd subjects)
                df_odd.loc[df_odd["FoveaOdd"] == 1, "FoveaOdd"] = "look"
                df_odd.loc[df_odd["FoveaOdd"] == 0, "FoveaOdd"] = "not look"

                ############################################### Even subject ###############################################
                # Combine all pre even files
                df_even = pd.read_csv(files_pre_even[idx], index_col=None, header=0)

                # Replace missing values with averages of columns where they are
                df_even.fillna(df_even.mean(), inplace=True)

                # Remove space before column names
                df_even_new_columns = df_even.columns.str.replace(" ", "")
                df_even.columns = df_even_new_columns

                # convert cartesian to degree of eye data

                # Gaze direction (right eye)
                df_even["GazeDirectionRight(X)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionRight(X)"], x["GazeDirectionRight(Y)"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionRight(Y)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionRight(Y)"], x["GazeDirectionRight(Z)"]
                    ),
                    axis=1,
                )

                # Gaze direction (left eye)
                df_even["GazeDirectionLeft(X)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionLeft(X)"], x["GazeDirectionLeft(Y)"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionLeft(Y)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionLeft(Y)"], x["GazeDirectionLeft(Z)"]
                    ),
                    axis=1,
                )

                # check degree_within_fovea or not
                df_even["GazeDirectionRight(X)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(X)Degree"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionRight(Y)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(Y)Degree"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionLeft(X)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(X)Degree"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionLeft(Y)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(Y)Degree"]
                    ),
                    axis=1,
                )

                # Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (even subjects)
                df_even["FoveaEven"] = (
                    df_even["GazeDirectionRight(X)inFovea"]
                    + df_even["GazeDirectionLeft(X)inFovea"]
                    + df_even["GazeDirectionRight(Y)inFovea"]
                    + df_even["GazeDirectionLeft(Y)inFovea"]
                )
                df_even.loc[df_even["FoveaEven"] > 1, "FoveaEven"] = 1

                # Change 1 => look , 0 => not look (even subjects)
                df_even.loc[df_even["FoveaEven"] == 1, "FoveaEven"] = "look"
                df_even.loc[df_even["FoveaEven"] == 0, "FoveaEven"] = "not look"

                # Calculate how many times they "really" look at each other
                looking_percentage_each_pair = (
                    self.__intermediaryEye.looking_percentage(df_odd, df_even)
                )

                # Put the percentage of looking each other of each pair into one list
                looking_percentage_all_pairs.append(looking_percentage_each_pair)

                indicator = str(idx + 1)
                end_info = "Pair-" + indicator + " " + tag + " is done"
                print(end_info)
                bar()

            return looking_percentage_all_pairs

    def diff_look_percent_pre_post(self, looking_percentage_all_pairs: list):
        """
        Objective  : Find difference of looking percentage between pre and post for three different eye conditions (averted, direct, and natural)

        Parameters :
                     - looking_percentage_all_pairs (list): Percentage of "really" looking at each other for all pairs for all eye conditions (both pre and post)
                        It is the output of eye_data_analysis function

        Outputs     :
                      Percentage of looking that has been deducted between pre and post for three different eye conditions:

                      diff_averted_eye,
                      diff_direct_eye,
                      diff_natural_eye
        """

        diff_averted_eye = [
            np.abs(x - y)
            for x, y in zip(
                looking_percentage_all_pairs[0], looking_percentage_all_pairs[1]
            )
        ]
        diff_direct_eye = [
            np.abs(x - y)
            for x, y in zip(
                looking_percentage_all_pairs[2], looking_percentage_all_pairs[3]
            )
        ]
        diff_natural_eye = [
            np.abs(x - y)
            for x, y in zip(
                looking_percentage_all_pairs[4], looking_percentage_all_pairs[5]
            )
        ]

        return diff_averted_eye, diff_direct_eye, diff_natural_eye

    def combine_eye_data(self, path2files: str, tag: str):

        """
        Objective  : Combine all cleaned eye tracker data for each eye condition, eg. averted_pre, into one dataframe.
                    It also involves pre-processing (replacing missing values with average value of columns
                    where they are). It calculates how much each pair looks at each other \n
                    throughout the experiment(each eye condition)

        Parameters : - path2files (str): Path to a directory where all cleaned eye tracker files are stored
                    - tag (str): eye gaze condition, ie. averted_pre, averted_post, direct_pre, direct_post, natural_pre, natural_post

        Output:
                    - combined_odd_df (dataframe)  : Combined dataframe of odd subjects (1, 3, 5 ..)
                    - combined_even_df (dataframe) : Combined dataframe of even subjects (2, 4, 6 ..)

        """

        gaze_keyword = "/*" + tag + "*.csv"
        pre_files = glob.glob(path2files + gaze_keyword)
        pattern = re.compile(r"[S]+(\d+)\-")
        files_pre_odd = []
        files_pre_even = []

        # Contain all odd dataframes
        odd_total_list = []
        # Contain all even dataframes
        even_total_list = []

        for idx, file in enumerate(pre_files):

            # Put into a list for ODD subjects - Refer to filename
            if (idx % 2) == 0:
                files_pre_odd.append(file)

            # Put into different list for EVEN subjects - Refer to filename
            else:
                files_pre_even.append(file)

        ############################################### Odd subject ###############################################
        # Combine all pre odd files

        with alive_bar(
            len(files_pre_odd), title="Eye Data(" + tag + ")", force_tty=True
        ) as bar:

            for idx, filename in enumerate(files_pre_odd):

                df_odd = pd.read_csv(filename, index_col=None, header=0)

                # Replace missing values with averages of columns where they are
                df_odd.fillna(df_odd.mean(), inplace=True)

                # Remove space before column names
                df_odd_new_columns = df_odd.columns.str.replace(" ", "")
                df_odd.columns = df_odd_new_columns

                # convert cartesian to degree of eye data

                # Gaze direction (right eye)
                df_odd["GazeDirectionRight(X)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionRight(X)"], x["GazeDirectionRight(Y)"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionRight(Y)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionRight(Y)"], x["GazeDirectionRight(Z)"]
                    ),
                    axis=1,
                )

                # Gaze direction (left eye)
                df_odd["GazeDirectionLeft(X)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionLeft(X)"], x["GazeDirectionLeft(Y)"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionLeft(Y)Degree"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionLeft(Y)"], x["GazeDirectionLeft(Z)"]
                    ),
                    axis=1,
                )

                # check degree_within_fovea or not
                df_odd["GazeDirectionRight(X)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(X)Degree"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionRight(Y)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(Y)Degree"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionLeft(X)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(X)Degree"]
                    ),
                    axis=1,
                )
                df_odd["GazeDirectionLeft(Y)inFovea"] = df_odd.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(Y)Degree"]
                    ),
                    axis=1,
                )

                # Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (odd subjects)
                df_odd["FoveaOdd"] = (
                    df_odd["GazeDirectionRight(X)inFovea"]
                    + df_odd["GazeDirectionLeft(X)inFovea"]
                    + df_odd["GazeDirectionRight(Y)inFovea"]
                    + df_odd["GazeDirectionLeft(Y)inFovea"]
                )
                df_odd.loc[df_odd["FoveaOdd"] > 1, "FoveaOdd"] = 1

                # Change 1 => look , 0 => not look (odd subjects)
                df_odd.loc[df_odd["FoveaOdd"] == 1, "FoveaOdd"] = "look"
                df_odd.loc[df_odd["FoveaOdd"] == 0, "FoveaOdd"] = "not look"

                # Populate all odd dataframes into a list
                odd_total_list.append(df_odd)

                ############################################### Even subject ###############################################
                # Combine all pre even files
                df_even = pd.read_csv(files_pre_even[idx], index_col=None, header=0)

                # Replace missing values with averages of columns where they are
                df_even.fillna(df_even.mean(), inplace=True)

                # Remove space before column names
                df_even_new_columns = df_even.columns.str.replace(" ", "")
                df_even.columns = df_even_new_columns

                # convert cartesian to degree of eye data

                # Gaze direction (right eye)
                df_even["GazeDirectionRight(X)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionRight(X)"], x["GazeDirectionRight(Y)"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionRight(Y)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionRight(Y)"], x["GazeDirectionRight(Z)"]
                    ),
                    axis=1,
                )

                # Gaze direction (left eye)
                df_even["GazeDirectionLeft(X)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_x_axis_to_degree(
                        x["GazeDirectionLeft(X)"], x["GazeDirectionLeft(Y)"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionLeft(Y)Degree"] = df_even.apply(
                    lambda x: self.__intermediaryEye.convert_gaze_direction_in_y_axis_to_degree(
                        x["GazeDirectionLeft(Y)"], x["GazeDirectionLeft(Z)"]
                    ),
                    axis=1,
                )

                # check degree_within_fovea or not
                df_even["GazeDirectionRight(X)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(X)Degree"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionRight(Y)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionRight(Y)Degree"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionLeft(X)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(X)Degree"]
                    ),
                    axis=1,
                )
                df_even["GazeDirectionLeft(Y)inFovea"] = df_even.apply(
                    lambda x: self.__intermediaryEye.check_degree_within_fovea(
                        x["GazeDirectionLeft(Y)Degree"]
                    ),
                    axis=1,
                )

                # Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (even subjects)
                df_even["FoveaEven"] = (
                    df_even["GazeDirectionRight(X)inFovea"]
                    + df_even["GazeDirectionLeft(X)inFovea"]
                    + df_even["GazeDirectionRight(Y)inFovea"]
                    + df_even["GazeDirectionLeft(Y)inFovea"]
                )
                df_even.loc[df_even["FoveaEven"] > 1, "FoveaEven"] = 1

                # Change 1 => look , 0 => not look (even subjects)
                df_even.loc[df_even["FoveaEven"] == 1, "FoveaEven"] = "look"
                df_even.loc[df_even["FoveaEven"] == 0, "FoveaEven"] = "not look"

                # Populate all even dataframes into a list
                even_total_list.append(df_even)

                indicator = str(idx + 1)
                end_info = "Pair-" + indicator + " " + tag + " is done"
                print(end_info)
                bar()

            # Combine all odd dataframes into a single dataframe
            combined_odd_df = pd.concat(odd_total_list).reset_index(drop=True)
            # Combine all even dataframes into a single dataframe
            combined_even_df = pd.concat(even_total_list).reset_index(drop=True)

        return combined_odd_df, combined_even_df

    def plot_heatmap(
        self,
        df_pre_odd: DataFrame,
        df_pre_even: DataFrame,
        df_post_odd: DataFrame,
        df_post_even: DataFrame,
        tag: str,
        save_fig=False,
        path2savefigure: str = os.getcwd(),
    ):

        """
        Objective  : Plot heatmap of gaze direction for a particular eye condition, i.e., averted, direct, and natural

        Parameters :
                    - df_pre_odd (DataFrame): Combined dataframe of pre training of odd subjects
                    - df_pre_even (DataFrame): Combined dataframe of pre training of even subjects
                    - df_post_odd (DataFrame): Combined dataframe of post training of odd subjects
                    - df_post_even (DataFrame): Combined dataframe of post training of even subjects
                                        NOTE : the above inputs (combined dataframe)
                                               is the output of combine_eye_data function

                    - tag (str) : Choose one of the options below. MUST be the exact word below :
                                    "averted", "direct", "natural"
                    - save_fig (boolean)=False (opt) : Option to save the figure. Default is not saving
                    - path2savefigure (str)=os.getcwd() (opt) : Set path to save the figure. Default is current working directory
        """

        odd_tag = tag.lower() + "_pre"
        even_tag = tag.lower() + "_post"

        # Prepare the data so that ready to be plotted

        # Change FoveaEven to Fovea and FoveaOdd to Fovea (Pre-training)
        df_pre_odd.rename(columns={"FoveaOdd": "Fovea"}, inplace=True)
        df_pre_even.rename(columns={"FoveaEven": "Fovea"}, inplace=True)

        # Change FoveaEven to Fovea and FoveaOdd to Fovea (Post-training)
        df_post_odd.rename(columns={"FoveaOdd": "Fovea"}, inplace=True)
        df_post_even.rename(columns={"FoveaEven": "Fovea"}, inplace=True)

        # Combine dataframe odd and even (Pre-training)
        df_combined_averted_pre = pd.concat(
            [df_pre_odd, df_pre_even], ignore_index=True
        )

        # Combine dataframe odd and even (Post-training)
        df_combined_averted_post = pd.concat(
            [df_post_odd, df_post_even], ignore_index=True
        )

        # Combine dataframe of Pre and Post-training
        df_combined_averted_pre_post = pd.concat(
            [df_combined_averted_pre, df_combined_averted_post], ignore_index=True
        )

        # Round off columns of GazeDirectionRight column
        df_combined_averted_pre_post = df_combined_averted_pre_post.round(
            {
                "GazeDirectionRight(X)": 1,
                "GazeDirectionRight(Y)": 1,
                "GazeDirectionLeft(X)": 1,
                "GazeDirectionLeft(Y)": 1,
            }
        )

        # Take only Gaze direction for both right and left eye
        df_combined_pre_post_right = df_combined_averted_pre_post[
            ["GazeDirectionRight(X)", "GazeDirectionRight(Y)"]
        ]
        df_combined_pre_post_left = df_combined_averted_pre_post[
            ["GazeDirectionLeft(X)", "GazeDirectionLeft(Y)"]
        ]

        # Rename the columns of left eye to be similar to right eye so that we can combine
        df_combined_pre_post_left.rename(
            columns={"GazeDirectionLeft(X)": "GazeDirectionRight(X)"}, inplace=True
        )
        df_combined_pre_post_left.rename(
            columns={"GazeDirectionLeft(Y)": "GazeDirectionRight(Y)"}, inplace=True
        )

        # Combine both right and left eye gaze directions into one dataframe (stacking each other - vertically)
        df_combined_averted_pre_post_gaze_only = pd.concat(
            [df_combined_pre_post_right, df_combined_pre_post_left], ignore_index=True
        )

        # Group multiple columns and reset the index
        df_group = (
            df_combined_averted_pre_post_gaze_only.groupby(
                ["GazeDirectionRight(X)", "GazeDirectionRight(Y)"]
            )
            .size()
            .reset_index(name="count")
        )
        # Count how many data that are under the same coordinate or spot
        pivot = df_group.pivot(
            index="GazeDirectionRight(X)",
            columns="GazeDirectionRight(Y)",
            values="count",
        )

        # Convert dataframe to numpy array
        pivot_array = pivot.to_numpy()

        # Plot the data with background picture

        fig, ax = plt.subplots(figsize=(7, 7))
        wd = matplotlib.cm.winter._segmentdata  # only has r,g,b
        wd["alpha"] = ((0.0, 0.0, 0.3), (0.3, 0.3, 1.0), (1.0, 1.0, 1.0))

        # modified colormap with changing alpha
        al_winter = LinearSegmentedColormap("AlphaWinter", wd)

        if tag == "averted":
            # get the overlay IMAGE as an array so we can plot it
            map_img = mpimg.imread("figures/averted2plot.png")
        elif tag == "direct":
            # get the overlay IMAGE as an array so we can plot it
            map_img = mpimg.imread("figures/direct2plot.png")
        elif tag == "natural":
            # get the overlay IMAGE as an array so we can plot it
            map_img = mpimg.imread("figures/natural2plot.png")

        # making and plotting heatmap

        colormap = sns.color_palette("Spectral", as_cmap=True)

        sns.set()

        # This is where we plot the data
        hmax = sns.heatmap(
            pivot_array,
            # cmap = al_winter, # this worked but I didn't like it
            cmap=colormap,
            alpha=0.48,  # whole heatmap is translucent
            annot=False,
            zorder=2,
            cbar_kws={"label": "Gaze-Direction Intensity"},
        )

        # heatmap uses pcolormesh instead of imshow, so we can't pass through
        # extent as a kwarg, so we can't mmatch the heatmap to the map. Instead,
        # match the map to the heatmap:

        # This is where we put the background picture
        hmax.imshow(
            map_img,
            aspect=hmax.get_aspect(),
            extent=hmax.get_xlim() + hmax.get_ylim(),
            zorder=1,
        )  # put the map under the heatmap

        # Make Capital the first letter of the tag
        tag_title = str.title(tag)
        # Combine as title of the figure
        tag_title = tag_title + "-Eye Gaze Condition"
        # Put as title
        ax.set_title(tag_title)

        ax.tick_params(axis="both", which="both", length=0)
        # Remove x labels
        ax.set(xticklabels=[], yticklabels=[])

        # Color bar related thing
        c_bar = hmax.collections[0].colorbar
        ticks_number = [ticks for ticks in c_bar.get_ticks()]
        c_bar.set_ticks([ticks_number[1], ticks_number[-2]])
        c_bar.set_ticklabels(["Low", "High"])

        # Save the figures
        if save_fig == True:

            fig = hmax.get_figure()

            # Save to the current working directory
            if path2savefigure == os.getcwd():
                cwd = os.getcwd()
                # os.chdir(cwd)
                file_name = tag + "_eye.png"
                cwd = os.path.join(cwd, file_name)
                fig.savefig(cwd, dpi=100)

            # Save to the desired path as mentioned in the parameter
            else:
                cwd = path2savefigure
                file_name = tag + "_eye.png"
                cwd = os.path.join(cwd, file_name)
                fig.savefig(cwd, dpi=100)
