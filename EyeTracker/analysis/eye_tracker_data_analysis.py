# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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

# %%
from math import atan, degrees
import re
import pandas as pd
import numpy as np
import random
import glob
import os
from pandas import DataFrame
from time import time
from datetime import timedelta
from sklearn.feature_selection import VarianceThreshold
# from tqdm import tqdm
from alive_progress import alive_bar
from alive_progress.styles import showtime


# %% [markdown]
# ## Function of eye_data_analysis 

# %%
def eye_data_analysis(path2files: str, tag:str):
    
    """
        Analyze all cleaned eye tracker data, eg. averted_pre.
        It also involves pre-processing (replacing missing values with average value of columns
        where they are). It calculates how much each pair looks at each other throughout the experiment(each eye condition)

    Args:
        path2files (str): Path to a directory where all cleaned eye tracker files are stored
        tag (str): eye gaze condition, ie. averted_pre, averted_post, direct_pre, direct_post, natural_pre, natural_post

    Returns:
        looking_percentage_all_pairs (list) : Each element represents the percentage of looking of each pair \n
                                              throughout the experiment
               
    """

    gaze_keyword= "/*" + tag + "*.csv"
    pre_files = glob.glob(path2files + gaze_keyword)
    pattern = re.compile(r"[S]+(\d+)\-")
    files_pre_odd = []
    files_pre_even = []
    looking_percentage_all_pairs = []

    for idx, file in enumerate(pre_files):
        # if int(re.search(pattern, file).group(1)) % 2 != 0:
        
        # Put into a list for ODD subjects - Refer to filename
        if ((idx  % 2 ) == 0):
            files_pre_odd.append(file)

        # Put into different list for EVEN subjects - Refer to filename
        else:
            files_pre_even.append(file)
    
    #TODO : Remove the line below when it is done
    # li_pre_odd = []
    # li_pre_even = []

    ############################################### Odd subject ###############################################
    # Combine all pre odd files
    
    with alive_bar(len(files_pre_odd), title="Eye Data(" + tag +")", force_tty=True) as bar:
        
        for idx, filename in enumerate(files_pre_odd):
            
            indicator = str(idx + 1)
            begin_info = "Pair-" + indicator + " is in progress..."
            print(begin_info)
                        
            df_odd = pd.read_csv(filename, index_col=None, header=0)
            # li_pre_odd.append(df_odd)
        
        #TODO : Remove the line below when it is done
        # # Populate all dataframes into one dataframe
        # df_odd = pd.concat(li_pre_odd, axis=0, ignore_index=True)

            # Replace missing values with averages of columns where they are
            df_odd.fillna(df_odd.mean(), inplace=True)

            #TODO : Remove the line below when it is done
            # df_odd = df_odd.reset_index(drop=True)

            # Remove space before column names
            df_odd_new_columns = df_odd.columns.str.replace(
                ' ', '')
            df_odd.columns = df_odd_new_columns

            # convert cartesian to degree of eye data
            
            # Gaze direction (right eye)
            df_odd['GazeDirectionRight(X)Degree'] = df_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
            df_odd['GazeDirectionRight(X)Degree'] = df_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
            df_odd['GazeDirectionRight(Y)Degree'] = df_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

            # Gaze direction (left eye)
            df_odd['GazeDirectionLeft(X)Degree'] = df_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
            df_odd['GazeDirectionLeft(Y)Degree'] = df_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)

            # check degree_within_fovea or not
            df_odd['GazeDirectionRight(X)inFovea'] = df_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
            df_odd['GazeDirectionRight(Y)inFovea'] = df_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
            df_odd['GazeDirectionLeft(X)inFovea'] = df_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
            df_odd['GazeDirectionLeft(Y)inFovea'] = df_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

            # Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (odd subjects)
            df_odd["FoveaOdd"] = df_odd["GazeDirectionRight(X)inFovea"] + df_odd["GazeDirectionLeft(X)inFovea"] + df_odd["GazeDirectionRight(Y)inFovea"] + df_odd["GazeDirectionLeft(Y)inFovea"]
            df_odd.loc[df_odd['FoveaOdd'] > 1,'FoveaOdd'] = 1

            # Change 1 => look , 0 => not look (odd subjects)
            df_odd.loc[df_odd['FoveaOdd'] == 1, 'FoveaOdd'] = 'look'
            df_odd.loc[df_odd['FoveaOdd'] == 0, 'FoveaOdd'] = 'not look'



        ############################################### Even subject ###############################################
        # Combine all pre even files
        # for filename in files_pre_even:
            # df_even = pd.read_csv(filename, index_col=None, header=0)
            df_even = pd.read_csv(files_pre_even[idx], index_col=None, header=0)
            # li_pre_even.append(df_even)
        
        #TODO : Remove the line below when it is done
        # Populate all dataframes into one dataframe
        # df_even = pd.concat(li_pre_even, axis=0, ignore_index=True)


            # Replace missing values with averages of columns where they are
            df_even.fillna(df_even.mean(), inplace=True)

            #TODO : Remove the line below when it is done
            # df_even = df_even.reset_index(drop=True)

            # Remove space before column names
            df_even_new_columns = df_even.columns.str.replace(
                ' ', '')
            df_even.columns = df_even_new_columns

            # convert cartesian to degree of eye data

            # Gaze direction (right eye)
            df_even['GazeDirectionRight(X)Degree'] = df_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
            df_even['GazeDirectionRight(X)Degree'] = df_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
            df_even['GazeDirectionRight(Y)Degree'] = df_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

            # Gaze direction (left eye)
            df_even['GazeDirectionLeft(X)Degree'] = df_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
            df_even['GazeDirectionLeft(Y)Degree'] = df_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)

            # check degree_within_fovea or not
            df_even['GazeDirectionRight(X)inFovea'] = df_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
            df_even['GazeDirectionRight(Y)inFovea'] = df_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
            df_even['GazeDirectionLeft(X)inFovea'] = df_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
            df_even['GazeDirectionLeft(Y)inFovea'] = df_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

            # Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (even subjects)
            df_even["FoveaEven"] = df_even["GazeDirectionRight(X)inFovea"] + df_even["GazeDirectionLeft(X)inFovea"] + df_even["GazeDirectionRight(Y)inFovea"] + df_even["GazeDirectionLeft(Y)inFovea"]
            df_even.loc[df_even['FoveaEven'] > 1,'FoveaEven'] = 1

            # Change 1 => look , 0 => not look (even subjects)
            df_even.loc[df_even['FoveaEven'] == 1,'FoveaEven'] = 'look'
            df_even.loc[df_even['FoveaEven'] ==0,'FoveaEven'] = 'not look'

        
            # Calculate how many times they "really" look at each other
            looking_percentage_each_pair = looking_percentage(df_odd, df_even)
            
            # Put the percentage of looking each other of each pair into one list
            looking_percentage_all_pairs.append(looking_percentage_each_pair)

            end_info = "Pair-" + indicator + " is done"      
            print(end_info)
            
            bar()

        return looking_percentage_all_pairs
        



##################################### Functions to convert to degree and check within fovea or not #################
### Put into a class of degree
# Formula to convert cartesian to degree
def gaze_direction_in_x_axis_degree(x: float, y: float):
    """Right hand rule coordinate"""
    try:
        degree = degrees(atan(y / x))  # Opposite / adjacent
    except ZeroDivisionError:
        degree = 0.0
    return round(degree, 2)

def gaze_direction_in_y_axis_degree(y: float, z: float):
    """Right hand rule coordinate"""
    try:
        degree = degrees(atan(z / y)) # Opposite / adjacent
    except ZeroDivisionError:
        degree = 0.0
    return round(degree, 2)

# Give mark 1 for GazeDirectionYDegree that falls under fovea are (30 degrees), otherwise 0
def check_degree_within_fovea(gaze_direction):
    """ In total 30 degrees where human can recognize an object. So we need to
    divide by 2. Half right and half left

    1 = within fovea
    0 = not wihtin fovea"""

    if (gaze_direction <= 15) & (gaze_direction >= 0):
        return 1
    elif (gaze_direction >= -15) & (gaze_direction <= 0):
        return 1
    else:
        return 0



###################### Function to check whether pair looks at each other or not ################### 
## Put into a class of analysis_hyperscanning

def looking_percentage(odd_dataframe:DataFrame, even_dataframe: DataFrame, srate:int =125, threshold:int =13):
    
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
        for j in range(i, i+srate):
            if odd_dataframe.iloc[j]['FoveaOdd']=='look' and even_dataframe.iloc[j]['FoveaEven']=='look':
                count += 1
        
        # each element of the list represents whether the pair looks at each other or not within a second
        list_look += [1 if count >= threshold else 0]

    # Percentage of looking
    percent_look = sum(list_look)/len(list_look) * 100
    # Get the last two digits only
    percent_look = float("{:.2f}".format(percent_look))

    return percent_look


            


# %% [markdown]
# ## Testing eye_data_analysis function
#

# %%
tag = "averted_pre"
path2files = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/raw_cleaned_combined_experimental_eye_data"
percent_looking_all = eye_data_analysis(path2files, tag)

# %% [markdown]
# ### Running a function of combine_eye_data_into_dataframe

# %%
tag = "averted_pre"
path2files = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/raw_cleaned_combined_experimental_eye_data"

df_averted = combine_eye_data_into_dataframe(path2files, tag)
print("averted_pre ODD subjects")
df_averted_pre_odd = df_averted[0]


print("averted_pre EVEN subjects")
df_averted_pre_even = df_averted[1]




# %% [markdown]
# ## Cut off the unneccessary columns of averted_pre EVEN subjects

# %%
# NOTE: No idea, why for some reasons the number of columns for df_averted_pre_even is almost double then df_averted_pre_odd dataframe
# So we need to cut it off for all EVEN subjects

# This is just to check the number of columns of odd subject
# odd_cols =  [x for x in df_averted_pre_odd.columns]
# print(len(odd_cols))

# # This is just to check the number of columns of even subject
# even_cols =  [x for x in df_averted_pre_even.columns]
# print(len(even_cols))

df_averted_pre_even = df_averted_pre_even.iloc[:,0:24]



# %% [markdown]
# ## Function to convert cartesian to degree (x and y)
# ## Function to check degree within fovea
# NOTE : Cartesian is a default value that is resulted from HTC Vive pro

# %%
# # Formula to convert cartesian to degree
# def gaze_direction_in_x_axis_degree(x: float, y: float):
#     """Right hand rule coordinate"""
#     try:
#         degree = degrees(atan(y / x))  # Opposite / adjacent
#     except ZeroDivisionError:
#         degree = 0.0
#     return round(degree, 2)

# def gaze_direction_in_y_axis_degree(y: float, z: float):
#     """Right hand rule coordinate"""
#     try:
#         degree = degrees(atan(z / y)) # Opposite / adjacent
#     except ZeroDivisionError:
#         degree = 0.0
#     return round(degree, 2)

# # Give mark 1 for GazeDirectionYDegree that falls under fovea are (30 degrees), otherwise 0
# def check_degree_within_fovea(gaze_direction):
#     """ In total 30 degrees where human can recognize an object. So we need to
#     divide by 2. Half right and half left

#     1 = within fovea
#     0 = not wihtin fovea"""

#     if (gaze_direction <= 15) & (gaze_direction >= 0):
#         return 1
#     elif (gaze_direction >= -15) & (gaze_direction <= 0):
#         return 1
#     else:
#         return 0

# %% [markdown]
# ### Running function to convert cartesian to degree (averted_pre_odd)

# %%
# Gaze direction (right eye)
df_averted_pre_odd['GazeDirectionRight(X)Degree'] = df_averted_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_pre_odd['GazeDirectionRight(X)Degree'] = df_averted_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_pre_odd['GazeDirectionRight(Y)Degree'] = df_averted_pre_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_averted_pre_odd['GazeDirectionLeft(X)Degree'] = df_averted_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_averted_pre_odd['GazeDirectionLeft(Y)Degree'] = df_averted_pre_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% [markdown]
# ### Running function to check_degree_within_fovea (averted_pre_odd)

# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_averted_pre_odd['GazeDirectionRight(X)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_averted_pre_odd['GazeDirectionRight(Y)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_averted_pre_odd['GazeDirectionLeft(X)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_averted_pre_odd['GazeDirectionLeft(Y)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

#df_averted_pre_odd.head(14874)

#df_averted_pre_odd.iloc[2, 'GazeDirectionRight(X)Degree'] 


# %%
# df_averted_pre_odd.head()

# %% [markdown]
# ### Running function to convert cartesian to degree (averted_pre_even)

# %%
# Gaze direction (right eye)
df_averted_pre_even['GazeDirectionRight(X)Degree'] = df_averted_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_pre_even['GazeDirectionRight(X)Degree'] = df_averted_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_pre_even['GazeDirectionRight(Y)Degree'] = df_averted_pre_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_averted_pre_even['GazeDirectionLeft(X)Degree'] = df_averted_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_averted_pre_even['GazeDirectionLeft(Y)Degree'] = df_averted_pre_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% [markdown]
# ### Running function to check_degree_within_fovea (averted_pre_even)

# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_averted_pre_even['GazeDirectionRight(X)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_averted_pre_even['GazeDirectionRight(Y)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_averted_pre_even['GazeDirectionLeft(X)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_averted_pre_even['GazeDirectionLeft(Y)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
# df_averted_pre_odd.columns

# %% [markdown]
# ## Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (odd subjects)

# %%

df_averted_pre_odd["FoveaOdd"] = df_averted_pre_odd["GazeDirectionRight(X)inFovea"] + df_averted_pre_odd["GazeDirectionLeft(X)inFovea"] + df_averted_pre_odd["GazeDirectionRight(Y)inFovea"] + df_averted_pre_odd["GazeDirectionLeft(Y)inFovea"]
df_averted_pre_odd.loc[df_averted_pre_odd['FoveaOdd'] > 1,'FoveaOdd'] = 1


# %% [markdown]
# ### Change 1 => look , 0 => not look (odd subjects)

# %%
df_averted_pre_odd.loc[df_averted_pre_odd['FoveaOdd'] == 1, 'FoveaOdd'] = 'look'
df_averted_pre_odd.loc[df_averted_pre_odd['FoveaOdd'] == 0, 'FoveaOdd'] = 'not look'


# %% [markdown]
# ## Compare values of in_fovea for both x-axis and y-axis whether both of them are 1 (even subjects)

# %%
df_averted_pre_even["FoveaEven"] = df_averted_pre_even["GazeDirectionRight(X)inFovea"] + df_averted_pre_even["GazeDirectionLeft(X)inFovea"] + df_averted_pre_even["GazeDirectionRight(Y)inFovea"] + df_averted_pre_even["GazeDirectionLeft(Y)inFovea"]
df_averted_pre_even.loc[df_averted_pre_even['FoveaEven'] > 1,'FoveaEven'] = 1

# %% [markdown]
# ### Change 1 => look , 0 => not look (even subjects)

# %%
df_averted_pre_even.loc[df_averted_pre_even['FoveaEven'] == 1,'FoveaEven'] = 'look'
df_averted_pre_even.loc[df_averted_pre_even['FoveaEven'] ==0,'FoveaEven'] = 'not look'

# %% [markdown]
# ## Checking if both participants are looking at each other or not - Not right the label at all
# If so, then add new column for both odd and even subjects "look_each_other" column. The value must be the same in both dataframes of columns "look_each_other"

# %%

# add looking_each_other to the original table for both odd and even subjects
# if df_averted_pre_odd['FoveaOdd'] == 'look' and df_averted_pre_even['FoveaEven']=='look':
#     df_averted_pre_odd['look_each_other'] = 1
#     df_averted_pre_even['look_each_other'] = `1
# else:
df_averted_pre_odd['look_each_other'] = np.where(df_averted_pre_odd['FoveaOdd'] == df_averted_pre_even['FoveaEven'], 1, 0) 
df_averted_pre_even['look_each_other'] = np.where(df_averted_pre_even['FoveaEven'] == df_averted_pre_odd['FoveaOdd'], 1, 0) 

# %%
# Checking if the number of look each other is the same or not. Just change odd to even
df_averted_pre_odd['look_each_other'].value_counts()

# %% [markdown]
# ## Counting how many times pair look each other

# %%
# def looking_percentage(odd_dataframe:DataFrame, even_dataframe: DataFrame, srate:int =125, threshold:int =13):
    
#     """ 
#         Objective  : Count how many times each pair "really" looks at each other throughout the experiment
#                      This will look at per second.  "Really" is determined by a previous research which indicates
#                      that humans are conscious look at each other within 100ms, 
#                      which is equal to there are 13 times of "looking" value within column of FoveaOdd or FoveaEven
        
#         Parameters : - odd_dataframe (pandas dataframe) :  dataframe of odd participant
#                      - even_dataframe (pandas dataframe) :  dataframe of even participant
#                      - srate (int) : which indicates the step / per second that we will check whether
#                                      the pair looking or not
#                      - threshold (int) : threshold to determine whether the pair "really" looks or not
#                                          if there are at least 13 times of "looking" within a second (srate)
#                                          under the column of FoveaOdd or FoveaEven, then it is considered "really" looking
#                         Note : Kindly refer to this research to see the threshold of 30 (100ms)
#                         https://journals.sagepub.com/doi/abs/10.1111/j.1467-9280.2006.01750.x?casa_token=AYU81Dg2DAMAAAAA%3Asy9nVGA6NjQPFuRthQW5eCZl9V06TpqV2OgtYbUFPwVKCV4so2PlVJrBWo01EfiSX-yNHul7mX_DlYk&journalCode=pssa

#         Output      : - percent_look (float) : percentage of looking of a pair throughout the experiment

#     """

#     list_look = []
#     # To chunk the data for every n raws (which indicate per second). Refer to srate
#     for i in range(0, odd_dataframe.shape[0], srate):
#         count = 0
#         # To loop the series or data in a dataframe so that we can check if they are matched or not
#         for j in range(i, i+srate):
#             if odd_dataframe.iloc[j]['FoveaOdd']=='look' and even_dataframe.iloc[j]['FoveaEven']=='look':
#                 count += 1
        
#         # each element of the list represents whether the pair looks at each other or not within a second
#         list_look += [1 if count >= threshold else 0]

#     # Percentage of looking
#     percent_look = sum(list_look)/len(list_look) * 100
#     # Get the last two digits only
#     percent_look = float("{:.2f}".format(percent_look))

#     return percent_look


# %%
percent_temp = looking_percentage(df_averted_pre_odd, df_averted_pre_even)
print(percent_temp)

# %%
percent_looking_all = []
# Put percentage of looking for each pair into a list so that we can use this later on
percent_looking_all.append(percent_look)

# Percentage of looking
percent_look = sum(list_look)/len(list_look) * 100
# Get the last two digits only
percent_look = float("{:.2f}".format(percent_look))



# %% [markdown]
# ## Give label how many "1" that is above threshold for every 125 rows (sampling rate) - This is odd dataframe
# Just do that for either odd or even dataframe
#
# ## This is just checking how many "1"s every 125 sampling rate (every second) - This is odd dataframe

# %%
threshold = 13
current_rows = 0 

while current_rows < len(df_averted_pre_odd) + 125:
    selection = df_averted_pre_odd.loc[current_rows: current_rows + 125, 'look_each_other']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    # Uncomment this to see how many "1"s for every 125 rows (second)
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_averted_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_averted_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_averted_pre_odd.loc[current_rows + 125: len(df_averted_pre_odd), 'PercentOdd'] = 0
df_averted_pre_odd.shape

# %% [markdown]
# ## Give label how many "1" that is above threshold for every 125 rows (sampling rate) - This is even dataframe
# Just do that for either odd or even dataframe. You don't need to run this when you have done the above cell. But it is ok for now
#
# ## This is just checking how many "1"s every 125 sampling rate (every second) - This is even dataframe

# %%
threshold = 13
current_rows = 0 

while current_rows < len(df_averted_pre_even) + 125:
    selection = df_averted_pre_even.loc[current_rows: current_rows + 125, 'look_each_other']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    # Uncomment this to see how many "1"s for every 125 rows (second)
    # print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_averted_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_averted_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
    

df_averted_pre_even.loc[current_rows + 125: len(df_averted_pre_even), 'PercentEven'] = 0



# %%
#TODO : Count how many "1" or percentage for every second (125)
df_averted_pre_odd["look_each_other"].shape

# %%
# df_averted_pre_even.drop(["sig_looking"],axis=1, inplace=True)
# df_averted_pre_even.columns

# %% [markdown]
# ## Give value 1 when the value of looking at each other above the threshold, which is 13, area available for both odd and even dataframe 
# If there are 13 times of "1" or more within one second that that is considered significant looking at each other are matched for both odd and even subjects. It ensures that they looked at each other "significantly"

# %%
df_averted_pre_odd['sig_looking'] = np.where((df_averted_pre_odd['PercentOdd'] == 1) & (df_averted_pre_even['PercentEven'] == 1), '1', '0')
df_averted_pre_even['sig_looking'] = np.where((df_averted_pre_odd['PercentOdd'] == 1) & (df_averted_pre_even['PercentEven'] == 1), '1', '0')

# %% [markdown]
# ## Count how many percentage of looking and not looking
# For averted condition, we could say 100% of participants did not look "scientifically" each other. Because it is zero

# %%
df_averted_pre_even['sig_looking'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 

# %%
col = df_averted_pre_odd["look_each_other"]
# The number of elements has to be 15 !!!!
seconds = [106, 116, 112, 107, 102, 118, 118, 102, 116, 111, 115, 108, 120, 113, 102]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 106, 222, 334, 441, 543, 661, 779, 895, 1006, 1121, 1229, 1349, 1462, 1564]
rows_each_pair_end =   [106, 222, 334, 441, 543, 661, 779, 895, 1006, 1121, 1229, 1349, 1462, 1564]

# Create new column for percentage looking and not looking
df_averted_pre_odd["look_percentage"]=""
df_averted_pre_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    # df_temp = df_averted_pre_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    df_temp = df_averted_pre_odd[rows_each_pair_begin[idx]:rows_each_pair_end[idx], :]

    # one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    # zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # # Calculate average
    # one_percentage = (one_counter / 125) * 100
    # zero_percentage = 100 - one_percentage
    # # Assign percentage on new column
    # df_averted_pre_odd.loc[idx, ["look_percentage"]] = one_percentage
    # df_averted_pre_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)


# %%
col = df_averted_pre_odd.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [106, 116, 112, 107, 102, 118, 118, 102, 116, 111, 115, 108, 120, 113, 102]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 106, 222, 334, 441, 543, 661, 779, 895, 1006, 1121, 1229, 1349, 1462, 1564]
rows_each_pair_end =   [106, 222, 334, 441, 543, 661, 779, 895, 1006, 1121, 1229, 1349, 1462, 1564]

# Create new column for percentage looking and not looking
df_averted_pre_odd["look_percentage"]=""
df_averted_pre_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_averted_pre_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_averted_pre_odd.loc[idx, ["look_percentage"]] = one_percentage
    df_averted_pre_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_averted_pre_odd

# %%
col = df_averted_pre_even.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [102, 116, 112, 107, 102, 106, 118, 118, 102, 116, 111, 115, 108, 120, 113]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 102, 218, 330, 437, 539, 645, 763, 881, 983, 1099, 1210, 1325, 1433, 1553, 1666]
rows_each_pair_end =   [102, 218, 330, 437, 539, 645, 763, 881, 983, 1099, 1210, 1325, 1433, 1553, 1666]

# Create new column for percentage looking and not looking
df_averted_pre_even["look_percentage"]=""
df_averted_pre_even["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_averted_pre_even.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_averted_pre_even.loc[idx, ["look_percentage"]] = one_percentage
    df_averted_pre_even.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_averted_pre_even

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new/'
averted_post_files = glob.glob(path + "/*averted_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
averted_files_post_odd = []
averted_files_post_even = []

for file in averted_post_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        averted_files_post_odd.append(file)
    else:
        averted_files_post_even.append(file)

li_averted_post_odd = []
li_averted_post_even = []

# ###############################################
# Combine all averted pre odd files
for filename in averted_files_post_odd:
    df_averted_post_odd = pd.read_csv(filename, index_col=None, header=0)
    li_averted_post_odd.append(df_averted_post_odd)
# Populate all dataframes into one dataframe
df_averted_post_odd = pd.concat(li_averted_post_odd, axis=0, ignore_index=True)
# Remove row where there is NaN value
df_averted_post_odd = df_averted_post_odd.dropna()
df_averted_post_odd = df_averted_post_odd.reset_index(drop=True)
# Remove space before column names
df_averted_post_odd_new_columns = df_averted_post_odd.columns.str.replace(
    ' ', '')
df_averted_post_odd.columns = df_averted_post_odd_new_columns
# df_averted_pre_odd.head()

# ###############################################
# Combine all averted pre even files
for filename in averted_files_post_even:
    df_averted_post_even = pd.read_csv(filename, index_col=None, header=0)
    li_averted_post_even.append(df_averted_post_even)

# Populate all dataframes into one dataframe
df_averted_post_even = pd.concat(li_averted_post_even, axis=0, ignore_index=True)

# Remove row where there is NaN value
df_averted_post_even = df_averted_post_even.dropna()
df_averted_post_even = df_averted_post_even.reset_index(drop=True)

# Remove space before column names
df_averted_post_even_new_columns = df_averted_post_even.columns.str.replace(
    ' ', '')
df_averted_post_even.columns = df_averted_post_even_new_columns
#df_averted_post_even.head()

# %%
# Gaze direction (right eye)
df_averted_post_odd['GazeDirectionRight(X)Degree'] = df_averted_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_post_odd['GazeDirectionRight(X)Degree'] = df_averted_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_post_odd['GazeDirectionRight(Y)Degree'] = df_averted_post_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_averted_post_odd['GazeDirectionLeft(X)Degree'] = df_averted_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_averted_post_odd['GazeDirectionLeft(Y)Degree'] = df_averted_post_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_averted_post_odd['GazeDirectionRight(X)inFovea'] = df_averted_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_averted_post_odd['GazeDirectionRight(Y)inFovea'] = df_averted_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_averted_post_odd['GazeDirectionLeft(X)inFovea'] = df_averted_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_averted_post_odd['GazeDirectionLeft(Y)inFovea'] = df_averted_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_averted_post_odd.head(14874)

# %%
#df_copy = df_averted_post_odd.copy()
#GazeDirectionRightDegree2 = df_copy.loc[:, 'GazeDirectionRight(X)Degree'] + df_copy.loc[:, 'GazeDirectionRight(Y)Degree']
#df_averted_pre_odd["in_fovea"]  
#GazeDirectionRightDegree2.to_frame().applymap(lambda x : check_degree_within_fovea(x) if ((x <= 15 and x >= 0) or (x >= -15 and x <= 0)) else 0)
#df_averted_post_odd.loc[df_averted_post_odd['GazeDirectionRight(X)inFovea'] <= 0,'FoveaOdd'] = 0
#df_averted_post_odd.loc[df_averted_post_odd['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaOdd'] = 0
#df_averted_post_odd.loc[df_averted_post_odd['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaOdd'] = 0
#df_averted_post_odd.loc[df_averted_post_odd['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaOdd'] = 0

df_averted_post_odd["FoveaOdd"] = df_averted_post_odd["GazeDirectionRight(X)inFovea"] + df_averted_post_odd["GazeDirectionLeft(X)inFovea"] + df_averted_post_odd["GazeDirectionRight(Y)inFovea"] + df_averted_post_odd["GazeDirectionLeft(Y)inFovea"]
df_averted_post_odd.loc[df_averted_post_odd['FoveaOdd'] >= 1,'FoveaOdd'] = 1

df_averted_post_odd

# %%
df_averted_post_odd.loc[df_averted_post_odd['FoveaOdd'] >= 1,'looking'] = 'look'
df_averted_post_odd.loc[df_averted_post_odd['FoveaOdd'] <=0,'looking'] = 'not look'
df_averted_post_odd

# %%
# Gaze direction (right eye)
df_averted_post_even['GazeDirectionRight(X)Degree'] = df_averted_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_post_even['GazeDirectionRight(X)Degree'] = df_averted_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_averted_post_even['GazeDirectionRight(Y)Degree'] = df_averted_post_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_averted_post_even['GazeDirectionLeft(X)Degree'] = df_averted_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_averted_post_even['GazeDirectionLeft(Y)Degree'] = df_averted_post_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_averted_post_even['GazeDirectionRight(X)inFovea'] = df_averted_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_averted_post_even['GazeDirectionRight(Y)inFovea'] = df_averted_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_averted_post_even['GazeDirectionLeft(X)inFovea'] = df_averted_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_averted_post_even['GazeDirectionLeft(Y)inFovea'] = df_averted_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
#df_averted_post_even.loc[df_averted_post_even['GazeDirectionRight(X)inFovea'] <= 0,'FoveaEven'] = 0
#df_averted_post_even.loc[df_averted_post_even['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaEven'] = 0
#df_averted_post_even.loc[df_averted_post_even['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaEven'] = 0
#df_averted_post_even.loc[df_averted_post_even['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaEven'] = 0

df_averted_post_even["FoveaEven"] = df_averted_post_even["GazeDirectionRight(X)inFovea"] + df_averted_post_even["GazeDirectionLeft(X)inFovea"] + df_averted_post_even["GazeDirectionRight(Y)inFovea"] + df_averted_post_even["GazeDirectionLeft(Y)inFovea"]
df_averted_post_even.loc[df_averted_post_even['FoveaEven'] >= 1,'FoveaEven'] = 1

df_averted_post_even

# %%
df_averted_post_even.loc[df_averted_post_even['FoveaEven'] >= 1,'looking'] = 'look'
df_averted_post_even.loc[df_averted_post_even['FoveaEven'] <=0,'looking'] = 'not look'
df_averted_post_even

# %%
df_averted_post_odd['look_each_other'] = np.where(df_averted_post_odd['FoveaOdd'] == df_averted_post_even['FoveaEven'], '1', '0') 
#create new column in df1 to check if  match
df_averted_post_odd

# %%
df_averted_post_even['look_each_other'] = np.where(df_averted_post_even['FoveaEven'] == df_averted_post_odd['FoveaOdd'], '1', '0') 
#create new column in df1 to check if  match
df_averted_post_even

# %%
df_averted_post_odd.loc[df_averted_post_odd.FoveaOdd <= 0, "look_each_other"] == "0"
df_averted_post_odd

# %%
df_averted_post_even.loc[df_averted_post_even.FoveaEven <= 0, "look_each_other"] = "0"
df_averted_post_even

# %%
df_averted_post_odd

# %%
df_averted_post_even

# %%
df_averted_post_odd['look_each_other'].head(120).value_counts()

# %%
df_averted_post_odd['look_each_other'].iloc[:120]

# %%
df_averted_post_odd['look_each_other'].head()

# %%
#Threshold = 13
#for index in df_averted_post_odd.index:
#    print(df_averted_post_odd['FoveaOdd'][index])
#def sanjit_algorithm(df_averted_post_odd, threshold=13, step_size=125, new_column_name="percent"):
#    df_averted_post_odd[new_column_name] = np.nan
#    for i in range(0, len(df_averted_post_odd), step_size):
#        condition = (df_averted_post_odd.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_averted_post_odd.iloc[i][new_column_name] = 1 if condition else 0
#    return df_averted_post_odd

#df_averted_post_odd = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_averted_post_odd = sanjit_algorithm(df_averted_post_odd)
#df_averted_post_odd

threshold = 13
current_rows = 0 

while current_rows < len(df_averted_post_odd) + 125:
    selection = df_averted_post_odd.loc[current_rows: current_rows + 125, 'FoveaOdd']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_averted_post_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_averted_post_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_averted_post_odd.loc[current_rows + 125: len(df_averted_post_odd), 'PercentOdd'] = 0

# %%
#def sanjit_algorithm(df_averted_post_even, threshold=13, step_size=125, new_column_name="percent"):
#    df_averted_post_even[new_column_name] = np.nan
#    for i in range(0, len(df_averted_post_even), step_size):
#        condition = (df_averted_post_even.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_averted_post_even.iloc[i][new_column_name] = 1 if condition else 0
#    return df_averted_post_even

#df_averted_post_even = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_averted_post_even = sanjit_algorithm(df_averted_post_even)
#df_averted_post_even

threshold = 13
current_rows = 0 

while current_rows < len(df_averted_post_even) + 125:
    selection = df_averted_post_even.loc[current_rows: current_rows + 125, 'FoveaEven']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_averted_post_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_averted_post_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
    
df_averted_post_even.loc[current_rows + 125: len(df_averted_post_odd), 'PercentEven'] = 0

# %%
df_averted_post_odd['ThresholdPercentage'] = np.where((df_averted_post_odd['PercentOdd'] == 1) & (df_averted_post_even['PercentEven'] == 1), '1', '0')
print (df_averted_post_odd.iloc[:1875])

# %%
df_averted_post_odd['ThresholdPercentage'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
averted_post_files = glob.glob(path + "/*averted_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
averted_files_post_odd = []
averted_files_post_even = []

for file in averted_post_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        averted_files_post_odd.append(file)
    else:
        averted_files_post_even.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(averted_files_post_odd):
    df_averted_post_odd = pd.read_csv(filename, index_col=None, header=0)
    sec = df_averted_post_odd.shape[0] / 125
    print(f"Averted_post_odd, pair : {idx}, Total Rows : {df_averted_post_odd.shape[0]}, seconds : {sec}")

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
averted_post_files = glob.glob(path + "/*averted_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
averted_files_post_odd = []
averted_files_post_even = []

for file in averted_post_files:
    if int(re.search(pattern, file).group(1)) % 2 == 0:
        averted_files_post_even.append(file)
    else:
        averted_files_post_odd.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(averted_files_post_even):
    df_averted_post_even = pd.read_csv(filename, index_col=None, header=0)
    sec = df_averted_post_even.shape[0] / 125
    print(f"Averted_post_even, pair : {idx}, Total Rows : {df_averted_post_even.shape[0]}, seconds : {sec}")

# %%
col = df_averted_post_odd.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [87, 120, 114, 104, 120, 111, 114, 110, 118, 105, 97, 103, 109, 120, 112]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 87, 207, 321, 425, 545, 656, 780, 890, 1008, 1113, 1210, 1313, 1422, 1542]
rows_each_pair_end =   [87, 207, 321, 425, 545, 656, 780, 890, 1008, 1113, 1210, 1313, 1422, 1542, 1654]

# Create new column for percentage looking and not looking
df_averted_post_odd["look_percentage"]=""
df_averted_post_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_averted_post_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_averted_post_odd.loc[idx, ["look_percentage"]] = one_percentage
    df_averted_post_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_averted_post_odd

# %%
col = df_averted_post_even.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [112, 120, 114, 104, 120, 87, 111, 114, 110, 118, 105, 97, 103, 109, 120]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 112, 232, 346, 450, 570, 657, 768, 882, 992, 1110, 1215, 1312, 1415, 1524]
rows_each_pair_end =   [112, 232, 346, 450, 570, 657, 768, 882, 992, 1110, 1215, 1312, 1415, 1524, 1644]

# Create new column for percentage looking and not looking
df_averted_post_even["look_percentage"]=""
df_averted_post_even["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_averted_post_even.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_averted_post_even.loc[idx, ["look_percentage"]] = one_percentage
    df_averted_post_even.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_averted_pre_even

# %%
#df_averted_pre_odd['look_each_other'].value_counts()
#df_averted_post_odd_v2 = df_averted_post_odd[['FoveaOdd']]
#df_averted_post_odd_v2

# %%
#df_averted_pre_even['look_each_other'].value_counts()
#df_averted_post_even_v2 = df_averted_pre_even[['FoveaEven']]
#df_averted_post_even_v2

# %%
#df_averted_pre_odd['look_each_other'].value_counts()
#df_even_odd_join2 = df_averted_post_even_v2.merge(df_averted_post_odd_v2, left_index=True, right_index=True, how='inner')
#df_even_odd_join2['look_match'] = [ x & y for x,y in zip(df_even_odd_join2['FoveaOdd'], df_even_odd_join2['FoveaEven'])]
#df_even_odd_join2['look_match'].value_counts()

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new/'
direct_pre_files = glob.glob(path + "/*direct_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
direct_files_pre_odd = []
direct_files_pre_even = []

for file in direct_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        direct_files_pre_odd.append(file)
    else:
        direct_files_pre_even.append(file)

li_direct_pre_odd = []
li_direct_pre_even = []

# ###############################################
# Combine all averted pre odd files
for filename in direct_files_pre_odd:
    df_direct_pre_odd = pd.read_csv(filename, index_col=None, header=0)
    li_direct_pre_odd.append(df_direct_pre_odd)
# Populate all dataframes into one dataframe
df_direct_pre_odd = pd.concat(li_direct_pre_odd, axis=0, ignore_index=True)
# Remove row where there is NaN value
df_direct_pre_odd = df_direct_pre_odd.dropna()
df_direct_pre_odd = df_direct_pre_odd.reset_index(drop=True)
# Remove space before column names
df_direct_pre_odd_new_columns = df_direct_pre_odd.columns.str.replace(
    ' ', '')
df_direct_pre_odd.columns = df_direct_pre_odd_new_columns
# df_averted_pre_odd.head()

# ###############################################
# Combine all averted pre even files
for filename in direct_files_pre_even:
    df_direct_pre_even = pd.read_csv(filename, index_col=None, header=0)
    li_direct_pre_even.append(df_direct_pre_even)

# Populate all dataframes into one dataframe
df_direct_pre_even = pd.concat(li_direct_pre_even, axis=0, ignore_index=True)

# Remove row where there is NaN value
df_direct_pre_even = df_direct_pre_even.dropna()
df_direct_pre_even = df_direct_pre_even.reset_index(drop=True)

# Remove space before column names
df_direct_pre_even_new_columns = df_direct_pre_even.columns.str.replace(
    ' ', '')
df_direct_pre_even.columns = df_direct_pre_even_new_columns
# df_averted_pre_even.head()

# %%
# Gaze direction (right eye)
df_direct_pre_odd['GazeDirectionRight(X)Degree'] = df_direct_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_pre_odd['GazeDirectionRight(X)Degree'] = df_direct_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_pre_odd['GazeDirectionRight(Y)Degree'] = df_direct_pre_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_direct_pre_odd['GazeDirectionLeft(X)Degree'] = df_direct_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_direct_pre_odd['GazeDirectionLeft(Y)Degree'] = df_direct_pre_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_direct_pre_odd['GazeDirectionRight(X)inFovea'] = df_direct_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_direct_pre_odd['GazeDirectionRight(Y)inFovea'] = df_direct_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_direct_pre_odd['GazeDirectionLeft(X)inFovea'] = df_direct_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_direct_pre_odd['GazeDirectionLeft(Y)inFovea'] = df_direct_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_direct_pre_odd.head(14874)

# %%
#df_copy = df_averted_post_odd.copy()
#GazeDirectionRightDegree2 = df_copy.loc[:, 'GazeDirectionRight(X)Degree'] + df_copy.loc[:, 'GazeDirectionRight(Y)Degree']
#df_averted_pre_odd["in_fovea"]  
#GazeDirectionRightDegree2.to_frame().applymap(lambda x : check_degree_within_fovea(x) if ((x <= 15 and x >= 0) or (x >= -15 and x <= 0)) else 0)
#df_direct_pre_odd.loc[df_direct_pre_odd['GazeDirectionRight(X)inFovea'] <= 0,'FoveaOdd'] = 0
#df_direct_pre_odd.loc[df_direct_pre_odd['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaOdd'] = 0
#df_direct_pre_odd.loc[df_direct_pre_odd['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaOdd'] = 0
#df_direct_pre_odd.loc[df_direct_pre_odd['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaOdd'] = 0

df_direct_pre_odd["FoveaOdd"] = df_direct_pre_odd["GazeDirectionRight(X)inFovea"] + df_direct_pre_odd["GazeDirectionLeft(X)inFovea"] + df_direct_pre_odd["GazeDirectionRight(Y)inFovea"] + df_direct_pre_odd["GazeDirectionLeft(Y)inFovea"]
df_direct_pre_odd.loc[df_direct_pre_odd['FoveaOdd'] >= 1,'FoveaOdd'] = 1

df_direct_pre_odd

# %%
df_direct_pre_odd.loc[df_direct_pre_odd['FoveaOdd'] >= 1,'looking'] = 'look'
df_direct_pre_odd.loc[df_direct_pre_odd['FoveaOdd'] <=0,'looking'] = 'not look'
df_direct_pre_odd

# %%
df_direct_pre_odd.head(20)

# %%
# Gaze direction (right eye)
df_direct_pre_even['GazeDirectionRight(X)Degree'] = df_direct_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_pre_even['GazeDirectionRight(X)Degree'] = df_direct_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_pre_even['GazeDirectionRight(Y)Degree'] = df_direct_pre_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_direct_pre_even['GazeDirectionLeft(X)Degree'] = df_direct_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_direct_pre_even['GazeDirectionLeft(Y)Degree'] = df_direct_pre_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_direct_pre_even['GazeDirectionRight(X)inFovea'] = df_direct_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_direct_pre_even['GazeDirectionRight(Y)inFovea'] = df_direct_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_direct_pre_even['GazeDirectionLeft(X)inFovea'] = df_direct_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_direct_pre_even['GazeDirectionLeft(Y)inFovea'] = df_direct_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
#df_direct_pre_even.loc[df_direct_pre_even['GazeDirectionRight(X)inFovea'] <= 0,'FoveaEven'] = 0
#df_direct_pre_even.loc[df_direct_pre_even['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaEven'] = 0
#df_direct_pre_even.loc[df_direct_pre_even['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaEven'] = 0
#df_direct_pre_even.loc[df_direct_pre_even['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaEven'] = 0

df_direct_pre_even["FoveaEven"] = df_direct_pre_even["GazeDirectionRight(X)inFovea"] + df_direct_pre_even["GazeDirectionLeft(X)inFovea"] + df_direct_pre_even["GazeDirectionRight(Y)inFovea"] + df_direct_pre_even["GazeDirectionLeft(Y)inFovea"]
df_direct_pre_even.loc[df_direct_pre_even['FoveaEven'] >= 1,'FoveaEven'] = 1

df_direct_pre_even

# %%
df_direct_pre_even.loc[df_direct_pre_even['FoveaEven'] >= 1,'looking'] = 'look'
df_direct_pre_even.loc[df_direct_pre_even['FoveaEven'] <=0,'looking'] = 'not look'
df_direct_pre_even

# %%
df_direct_pre_odd['look_each_other'] = np.where(df_direct_pre_odd['FoveaOdd'] == df_direct_pre_even['FoveaEven'], '1', '0') 
#create new column in df1 to check if  match
df_direct_pre_odd

# %%
df_direct_pre_even['look_each_other'] = np.where(df_direct_pre_even['FoveaEven'] == df_direct_pre_odd['FoveaOdd'], '1', '0') 
#create new column in df1 to check if  match
df_direct_pre_even

# %%
df_direct_pre_odd.loc[df_direct_pre_odd.FoveaOdd <= 0, "look_each_other"] = "0"
df_direct_pre_odd

# %%
df_direct_pre_even.loc[df_direct_pre_even.FoveaEven <= 0, "look_each_other"] = "0"
df_direct_pre_even

# %%
df_direct_pre_odd['look_each_other'].head(120).value_counts()

# %%
df_direct_pre_odd['look_each_other'].iloc[:120]

# %%
df_direct_pre_even

# %%
#Threshold = 13
#for index in df_direct_pre_odd.index:
#    print(df_direct_pre_odd['FoveaOdd'][index])\
#def sanjit_algorithm(df_direct_pre_odd, threshold=13, step_size=125, new_column_name="percent"):
#    df_direct_pre_odd[new_column_name] = np.nan
#    for i in range(0, len(df_direct_pre_odd), step_size):
#        condition = (df_direct_pre_odd.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_direct_pre_odd.iloc[i][new_column_name] = 1 if condition else 0
#    return df_direct_pre_odd

#df_direct_pre_odd = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_direct_pre_odd = sanjit_algorithm(df_direct_pre_odd)
#df_direct_pre_odd

threshold = 13
current_rows = 0 

while current_rows < len(df_direct_pre_odd) + 125:
    selection = df_direct_pre_odd.loc[current_rows: current_rows + 125, 'FoveaOdd']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_direct_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_direct_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_direct_pre_odd.loc[current_rows + 125: len(df_direct_pre_odd), 'PercentOdd'] = 0


# %%
#Threshold = 13
#for index in df_direct_pre_even.index:
#    print(df_direct_pre_even['FoveaEven'][index])
#def sanjit_algorithm(df_direct_pre_even, threshold=13, step_size=125, new_column_name="percent"):
#    df_direct_pre_even[new_column_name] = np.nan
#    for i in range(0, len(df_direct_pre_even), step_size):
#        condition = (df_direct_pre_even.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_direct_pre_even.iloc[i][new_column_name] = 1 if condition else 0
#    return df_direct_pre_even

#df_direct_pre_even = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_direct_pre_even = sanjit_algorithm(df_direct_pre_even)
#df_direct_pre_even

threshold = 13
current_rows = 0 

while current_rows < len(df_direct_pre_even) + 125:
    selection = df_direct_pre_even.loc[current_rows: current_rows + 125, 'FoveaEven']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_direct_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_direct_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
        
df_direct_pre_even.loc[current_rows + 125: len(df_direct_pre_even), 'PercentEven'] = 0



# %%
df_direct_pre_odd['ThresholdPercentage'] = np.where((df_direct_pre_odd['PercentOdd'] == 1) & (df_direct_pre_even['PercentEven'] == 1), '1', '0')
print (df_direct_pre_odd.iloc[:1875])

# %%
df_direct_pre_odd['ThresholdPercentage'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
direct_pre_files = glob.glob(path + "/*direct_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
direct_files_pre_odd = []
direct_files_pre_even = []

for file in direct_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        direct_files_pre_odd.append(file)
    else:
        direct_files_pre_even.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(direct_files_pre_odd):
    df_direct_pre_odd = pd.read_csv(filename, index_col=None, header=0)
    sec = df_direct_pre_odd.shape[0] / 125
    print(f"Direct_pre_odd, pair : {idx}, Total Rows : {df_direct_pre_odd.shape[0]}, seconds : {sec}")

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
direct_pre_files = glob.glob(path + "/*direct_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
direct_files_pre_odd = []
direct_files_pre_even = []

for file in direct_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 == 0:
        direct_files_pre_even.append(file)
    else:
        direct_files_pre_odd.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(direct_files_pre_even):
    df_direct_pre_even = pd.read_csv(filename, index_col=None, header=0)
    sec = df_direct_pre_even.shape[0] / 125
    print(f"Direct_pre_even, pair : {idx}, Total Rows : {df_direct_pre_even.shape[0]}, seconds : {sec}")

# %%
col = df_direct_pre_odd.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [91, 110, 108, 106, 114, 116, 117, 106, 119, 108, 102, 120, 107, 120, 106]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 91, 201, 309, 415, 519, 625, 742, 848, 967, 1075, 1177, 1297, 1404, 1524]
rows_each_pair_end =   [91, 201, 309, 415, 519, 625, 742, 848, 967, 1075, 1177, 1297, 1404, 1524, 1630]

# Create new column for percentage looking and not looking
df_direct_pre_odd["look_percentage"]=""
df_direct_pre_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_direct_pre_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_direct_pre_odd.loc[idx, ["look_percentage"]] = one_percentage
    df_direct_pre_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_direct_pre_odd

# %%
col = df_direct_pre_even.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [106, 110, 108, 106, 114, 91, 116, 117, 106, 119, 108, 102, 120, 107, 120]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 106, 216, 324, 430, 544, 635, 751, 868, 974, 1093, 1201, 1303, 1423, 1530]
rows_each_pair_end =   [106, 216, 324, 430, 544, 635, 751, 868, 974, 1093, 1201, 1303, 1423, 1530, 1650]

# Create new column for percentage looking and not looking
df_direct_pre_even["look_percentage"]=""
df_direct_pre_even["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_direct_pre_even.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_direct_pre_even.loc[idx, ["look_percentage"]] = one_percentage
    df_direct_pre_even.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_direct_pre_even

# %%
#df_averted_pre_odd['look_each_other'].value_counts()
#df_direct_pre_odd_v2 = df_direct_pre_odd[['FoveaOdd']]
#df_direct_pre_odd_v2

# %%
#df_direct_pre_even_v2 = df_direct_pre_even[['FoveaEven']]
#df_direct_pre_even_v2

# %%
#df_averted_pre_odd['look_each_other'].value_counts()
#df_even_odd_join3 = df_direct_pre_even_v2.merge(df_direct_pre_odd_v2, left_index=True, right_index=True, how='inner')
#df_even_odd_join3['look_match'] = [ x & y for x,y in zip(df_even_odd_join3['FoveaOdd'], df_even_odd_join3['FoveaEven'])]
#df_even_odd_join3['look_match'].value_counts()

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new/'
direct_post_files = glob.glob(path + "/*direct_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
direct_files_post_odd = []
direct_files_post_even = []

for file in direct_post_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        direct_files_post_odd.append(file)
    else:
        direct_files_post_even.append(file)

li_direct_post_odd = []
li_direct_post_even = []

# ###############################################
# Combine all averted pre odd files
for filename in direct_files_post_odd:
    df_direct_post_odd = pd.read_csv(filename, index_col=None, header=0)
    li_direct_post_odd.append(df_direct_post_odd)
# Populate all dataframes into one dataframe
df_direct_post_odd = pd.concat(li_direct_post_odd, axis=0, ignore_index=True)
# Remove row where there is NaN value
df_direct_post_odd = df_direct_post_odd.dropna()
df_direct_post_odd = df_direct_post_odd.reset_index(drop=True)
# Remove space before column names
df_direct_post_odd_new_columns = df_direct_post_odd.columns.str.replace(
    ' ', '')
df_direct_post_odd.columns = df_direct_post_odd_new_columns
# df_averted_pre_odd.head()

# ###############################################
# Combine all averted pre even files
for filename in direct_files_post_even:
    df_direct_post_even = pd.read_csv(filename, index_col=None, header=0)
    li_direct_post_even.append(df_direct_post_even)

# Populate all dataframes into one dataframe
df_direct_post_even = pd.concat(li_direct_post_even, axis=0, ignore_index=True)

# Remove row where there is NaN value
df_direct_post_even = df_direct_post_even.dropna()
df_direct_post_even = df_direct_post_even.reset_index(drop=True)

# Remove space before column names
df_direct_post_even_new_columns = df_direct_post_even.columns.str.replace(
    ' ', '')
df_direct_post_even.columns = df_direct_post_even_new_columns
# df_averted_pre_even.head()

# %%
# Gaze direction (right eye)
df_direct_post_odd['GazeDirectionRight(X)Degree'] = df_direct_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_post_odd['GazeDirectionRight(X)Degree'] = df_direct_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_post_odd['GazeDirectionRight(Y)Degree'] = df_direct_post_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_direct_post_odd['GazeDirectionLeft(X)Degree'] = df_direct_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_direct_post_odd['GazeDirectionLeft(Y)Degree'] = df_direct_post_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_direct_post_odd['GazeDirectionRight(X)inFovea'] = df_direct_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_direct_post_odd['GazeDirectionRight(Y)inFovea'] = df_direct_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_direct_post_odd['GazeDirectionLeft(X)inFovea'] = df_direct_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_direct_post_odd['GazeDirectionLeft(Y)inFovea'] = df_direct_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_direct_post_odd.head(14874)

# %%
#df_copy = df_direct_post_odd.copy()
#GazeDirectionRightDegree4 = df_copy.loc[:, 'GazeDirectionRight(X)Degree'] + df_copy.loc[:, 'GazeDirectionRight(Y)Degree']
#df_averted_pre_odd["in_fovea"]  
#GazeDirectionRightDegree4.to_frame().applymap(lambda x : check_degree_within_fovea(x) if ((x <= 15 and x >= 0) or (x >= -15 and x <= 0)) else 0)
#df_direct_post_odd.loc[df_direct_post_odd['GazeDirectionRight(X)inFovea'] <= 0,'FoveaOdd'] = 0
#df_direct_post_odd.loc[df_direct_post_odd['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaOdd'] = 0
#df_direct_post_odd.loc[df_direct_post_odd['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaOdd'] = 0
#df_direct_post_odd.loc[df_direct_post_odd['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaOdd'] = 0

df_direct_post_odd["FoveaOdd"] = df_direct_post_odd["GazeDirectionRight(X)inFovea"] + df_direct_post_odd["GazeDirectionLeft(X)inFovea"] + df_direct_post_odd["GazeDirectionRight(Y)inFovea"] + df_direct_post_odd["GazeDirectionLeft(Y)inFovea"]
df_direct_post_odd.loc[df_direct_post_odd['FoveaOdd'] >= 1,'FoveaOdd'] = 1

df_direct_post_odd

# %%
df_direct_post_odd.loc[df_direct_post_odd['FoveaOdd'] >= 1,'looking'] = 'look'
df_direct_post_odd.loc[df_direct_post_odd['FoveaOdd'] <=0,'looking'] = 'not look'
df_direct_post_odd

# %%
#df_copy = df_direct_post_even.copy()
#GazeDirectionLeftDegree4 = df_copy.loc[:, 'GazeDirectionLeft(X)Degree'] + df_copy.loc[:, 'GazeDirectionLeft(Y)Degree']
#df_averted_pre_odd["in_fovea"]  
#GazeDirectionLeftDegree4.to_frame().applymap(lambda x : check_degree_within_fovea(x) if ((x <= 15 and x >= 0) or (x >= -15 and x <= 0)) else 0)

# Gaze direction (right eye)
df_direct_post_even['GazeDirectionRight(X)Degree'] = df_direct_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_post_even['GazeDirectionRight(X)Degree'] = df_direct_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_direct_post_even['GazeDirectionRight(Y)Degree'] = df_direct_post_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_direct_post_even['GazeDirectionLeft(X)Degree'] = df_direct_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_direct_post_even['GazeDirectionLeft(Y)Degree'] = df_direct_post_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_direct_post_even['GazeDirectionRight(X)inFovea'] = df_direct_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_direct_post_even['GazeDirectionRight(Y)inFovea'] = df_direct_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_direct_post_even['GazeDirectionLeft(X)inFovea'] = df_direct_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_direct_post_even['GazeDirectionLeft(Y)inFovea'] = df_direct_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
#df_direct_post_even.loc[df_direct_post_even['GazeDirectionRight(X)inFovea'] <= 0,'FoveaEven'] = 0
#df_direct_post_even.loc[df_direct_post_even['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaEven'] = 0
#df_direct_post_even.loc[df_direct_post_even['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaEven'] = 0
#df_direct_post_even.loc[df_direct_post_even['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaEven'] = 0

df_direct_post_even["FoveaEven"] = df_direct_post_even["GazeDirectionRight(X)inFovea"] + df_direct_post_even["GazeDirectionLeft(X)inFovea"] + df_direct_post_even["GazeDirectionRight(Y)inFovea"] + df_direct_post_even["GazeDirectionLeft(Y)inFovea"]
df_direct_post_even.loc[df_direct_post_even['FoveaEven'] >= 1,'FoveaEven'] = 1
df_direct_post_even

# %%
df_direct_post_even.loc[df_direct_post_even['FoveaEven'] >= 1,'looking'] = 'look'
df_direct_post_even.loc[df_direct_post_even['FoveaEven'] <=0,'looking'] = 'not look'
df_direct_post_even

# %%
df_direct_post_odd['look_each_other'] = np.where(df_direct_post_odd['FoveaOdd'] == df_direct_post_even['FoveaEven'], '1', '0') 
#create new column in df1 to check if  match
df_direct_pre_odd

# %%
df_direct_post_even['look_each_other'] = np.where(df_direct_post_even['FoveaEven'] == df_direct_post_odd['FoveaOdd'], '1', '0') 
#create new column in df1 to check if  match
df_direct_post_even

# %%
df_direct_post_odd.loc[df_direct_post_odd.FoveaOdd <= 0, "look_each_other"] = "0"
df_direct_post_odd

# %%
df_direct_post_even.loc[df_direct_post_even.FoveaEven <= 0, "look_each_other"] = "0"
df_direct_post_even

# %%
df_direct_post_odd

# %%
df_direct_post_even

# %%
df_direct_post_odd['look_each_other'].head(120).value_counts()

# %%
df_direct_post_odd['look_each_other'].iloc[:120]

# %%
#hreshold = 13
#for index in df_direct_post_odd.index:
#    print(df_direct_post_odd['FoveaOdd'][index])
 
#def sanjit_algorithm(df_direct_post_odd, threshold=13, step_size=125, new_column_name="percent"):
#    df_direct_post_odd[new_column_name] = np.nan
#    for i in range(0, len(df_direct_post_odd), step_size):
#        condition = (df_direct_post_odd.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_direct_post_odd.iloc[i][new_column_name] = 1 if condition else 0
#    return df_direct_post_odd

#df_direct_post_odd = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_direct_post_odd = sanjit_algorithm(df_direct_post_odd)
#df_direct_post_odd

threshold = 13
current_rows = 0 

while current_rows < len(df_direct_post_odd) + 125:
    selection = df_direct_post_odd.loc[current_rows: current_rows + 125, 'FoveaOdd']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_direct_post_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_direct_post_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_direct_post_odd.loc[current_rows + 125: len(df_direct_post_odd), 'PercentOdd'] = 0


# %%
#Threshold = 13
#for index in df_direct_post_even.index:
#    print(df_direct_post_even['FoveaEven'][index])

#def sanjit_algorithm(df_direct_post_even, threshold=13, step_size=125, new_column_name="percent"):
#    df_direct_post_even[new_column_name] = np.nan
#    for i in range(0, len(df_direct_post_even), step_size):
#        condition = (df_direct_post_even.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_direct_post_even.iloc[i][new_column_name] = 1 if condition else 0
#    return df_direct_post_even

#df_direct_post_even = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_direct_post_even = sanjit_algorithm(df_direct_post_even)
#df_direct_post_even

threshold = 13
current_rows = 0 

while current_rows < len(df_direct_post_even) + 125:
    selection = df_direct_post_even.loc[current_rows: current_rows + 125, 'FoveaEven']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_direct_post_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_direct_post_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
        
df_direct_post_even.loc[current_rows + 125: len(df_direct_post_even), 'PercentEven'] = 0


# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
direct_post_files = glob.glob(path + "/*direct_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
direct_files_post_odd = []
direct_files_post_even = []

for file in direct_post_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        direct_files_post_odd.append(file)
    else:
        direct_files_post_even.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(direct_files_post_odd):
    df_direct_post_odd = pd.read_csv(filename, index_col=None, header=0)
    sec = df_direct_post_odd.shape[0] / 125
    print(f"Direct_post_odd, pair : {idx}, Total Rows : {df_direct_post_odd.shape[0]}, seconds : {sec}")

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
direct_post_files = glob.glob(path + "/*direct_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
direct_files_post_odd = []
direct_files_post_even = []

for file in direct_post_files:
    if int(re.search(pattern, file).group(1)) % 2 == 0:
        direct_files_post_even.append(file)
    else:
        direct_files_post_odd.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(direct_files_post_even):
    df_direct_post_even = pd.read_csv(filename, index_col=None, header=0)
    sec = df_direct_post_even.shape[0] / 125
    print(f"Direct_post_even, pair : {idx}, Total Rows : {df_direct_post_even.shape[0]}, seconds : {sec}")

# %%
col = df_direct_post_odd.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [90, 120, 117, 96, 106, 109, 115, 94, 120, 94, 94, 105, 120, 117, 103]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 90, 210, 327, 423, 529, 638, 753, 847, 967, 1061, 1155, 1260, 1380, 1497]
rows_each_pair_end =   [90, 210, 327, 423, 529, 638, 753, 847, 967, 1061, 1155, 1260, 1380, 1497, 1680]

# Create new column for percentage looking and not looking
df_direct_post_odd["look_percentage"]=""
df_direct_post_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_direct_post_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_direct_post_odd.loc[idx, ["look_percentage"]] = one_percentage
    df_direct_post_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_direct_post_odd

# %%
col = df_direct_post_even.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [103, 120, 117, 96, 106, 90, 109, 115, 94, 120, 94, 94, 105, 120, 117]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 103, 223, 340, 436, 542, 632, 741, 856, 950, 1070, 1164, 1258, 1363, 1483]
rows_each_pair_end =   [103, 223, 340, 436, 542, 632, 741, 856, 950, 1070, 1164, 1258, 1363, 1483, 1600]

# Create new column for percentage looking and not looking
df_direct_post_even["look_percentage"]=""
df_direct_post_even["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_direct_post_even.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_direct_post_even.loc[idx, ["look_percentage"]] = one_percentage
    df_direct_post_even.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_direct_post_even

# %%
df_direct_post_odd['ThresholdPercentage'] = np.where((df_direct_post_odd['PercentOdd'] == 1) & (df_direct_post_even['PercentEven'] == 1), '1', '0')
print (df_direct_post_odd.iloc[:1875])

# %%
df_direct_post_odd['ThresholdPercentage'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 

# %%
#df_direct_post_odd_v2 = df_direct_post_odd[['FoveaOdd']]
#df_direct_post_odd_v2

# %%
#df_direct_post_even_v2 = df_direct_post_even[['FoveaEven']]
#df_direct_post_even_v2

# %%
#df_even_odd_join4 = df_direct_post_even_v2.merge(df_direct_post_odd_v2, left_index=True, right_index=True, how='inner')
#df_even_odd_join4['look_match'] = [ x & y for x,y in zip(df_even_odd_join4['FoveaOdd'], df_even_odd_join4['FoveaEven'])]
#df_even_odd_join4['look_match'].value_counts()

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new/'
natural_pre_files = glob.glob(path + "/*natural_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
natural_files_pre_odd = []
natural_files_pre_even = []

for file in natural_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        natural_files_pre_odd.append(file)
    else:
        natural_files_pre_even.append(file)

li_natural_pre_odd = []
li_natural_pre_even = []

# ###############################################
# Combine all averted pre odd files
for filename in natural_files_pre_odd:
    df_natural_pre_odd = pd.read_csv(filename, index_col=None, header=0)
    li_natural_pre_odd.append(df_natural_pre_odd)
# Populate all dataframes into one dataframe
df_natural_pre_odd = pd.concat(li_natural_pre_odd, axis=0, ignore_index=True)
# Remove row where there is NaN value
df_natural_pre_odd = df_natural_pre_odd.dropna()
df_natural_pre_odd = df_natural_pre_odd.reset_index(drop=True)
# Remove space before column names
df_natural_pre_odd_new_columns = df_natural_pre_odd.columns.str.replace(
    ' ', '')
df_natural_pre_odd.columns = df_natural_pre_odd_new_columns
# df_averted_pre_odd.head()

# ###############################################
# Combine all averted pre even files
for filename in natural_files_pre_even:
    df_natural_pre_even = pd.read_csv(filename, index_col=None, header=0)
    li_natural_pre_even.append(df_natural_pre_even)

# Populate all dataframes into one dataframe
df_natural_pre_even = pd.concat(li_natural_pre_even, axis=0, ignore_index=True)

# Remove row where there is NaN value
df_natural_pre_even = df_natural_pre_even.dropna()
df_natural_pre_even = df_natural_pre_even.reset_index(drop=True)

# Remove space before column names
df_natural_pre_even_new_columns = df_natural_pre_even.columns.str.replace(
    ' ', '')
df_natural_pre_even.columns = df_natural_pre_even_new_columns
# df_averted_pre_even.head()

# %%
# Gaze direction (right eye)
df_natural_pre_odd['GazeDirectionRight(X)Degree'] = df_natural_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_pre_odd['GazeDirectionRight(X)Degree'] = df_natural_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_pre_odd['GazeDirectionRight(Y)Degree'] = df_natural_pre_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_natural_pre_odd['GazeDirectionLeft(X)Degree'] = df_natural_pre_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_natural_pre_odd['GazeDirectionLeft(Y)Degree'] = df_natural_pre_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_natural_pre_odd['GazeDirectionRight(X)inFovea'] = df_natural_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_natural_pre_odd['GazeDirectionRight(Y)inFovea'] = df_natural_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_natural_pre_odd['GazeDirectionLeft(X)inFovea'] = df_natural_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_natural_pre_odd['GazeDirectionLeft(Y)inFovea'] = df_natural_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_natural_pre_odd.head(14874)

# %%
#df_natural_pre_odd.loc[df_natural_pre_odd['GazeDirectionRight(X)inFovea'] <= 0,'FoveaOdd'] = 0
#df_natural_pre_odd.loc[df_natural_pre_odd['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaOdd'] = 0
#df_natural_pre_odd.loc[df_natural_pre_odd['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaOdd'] = 0
#df_natural_pre_odd.loc[df_natural_pre_odd['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaOdd'] = 0

df_natural_pre_odd["FoveaOdd"] = df_natural_pre_odd["GazeDirectionRight(X)inFovea"] + df_natural_pre_odd["GazeDirectionLeft(X)inFovea"] + df_natural_pre_odd["GazeDirectionRight(Y)inFovea"] + df_natural_pre_odd["GazeDirectionLeft(Y)inFovea"]
df_natural_pre_odd.loc[df_natural_pre_odd['FoveaOdd'] >= 1,'FoveaOdd'] = 1

df_natural_pre_odd

# %%
df_natural_pre_odd.loc[df_natural_pre_odd['FoveaOdd'] >= 1,'looking'] = 'look'
df_natural_pre_odd.loc[df_natural_pre_odd['FoveaOdd'] <=0,'looking'] = 'not look'
df_natural_pre_odd

# %%
# Gaze direction (right eye)
df_natural_pre_even['GazeDirectionRight(X)Degree'] = df_natural_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_pre_even['GazeDirectionRight(X)Degree'] = df_natural_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_pre_even['GazeDirectionRight(Y)Degree'] = df_natural_pre_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_natural_pre_even['GazeDirectionLeft(X)Degree'] = df_natural_pre_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_natural_pre_even['GazeDirectionLeft(Y)Degree'] = df_natural_pre_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_natural_pre_even['GazeDirectionRight(X)inFovea'] = df_natural_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_natural_pre_even['GazeDirectionRight(Y)inFovea'] = df_natural_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_natural_pre_even['GazeDirectionLeft(X)inFovea'] = df_natural_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_natural_pre_even['GazeDirectionLeft(Y)inFovea'] = df_natural_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
#df_natural_pre_even.loc[df_natural_pre_even['GazeDirectionRight(X)inFovea'] <= 0,'FoveaEven'] = 0
#df_natural_pre_even.loc[df_natural_pre_even['GazeDirectionLeft(X)inFovea']  <=1, 'FoveaEven'] = 0
#df_natural_pre_even.loc[df_natural_pre_even['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaEven'] = 0
#df_natural_pre_even.loc[df_natural_pre_even['GazeDirectionLeft(Y)inFovea']  <=1, 'FoveaEven'] = 0

df_natural_pre_even["FoveaEven"] = df_natural_pre_even["GazeDirectionRight(X)inFovea"] + df_natural_pre_even["GazeDirectionLeft(X)inFovea"] + df_natural_pre_even["GazeDirectionRight(Y)inFovea"] + df_natural_pre_even["GazeDirectionLeft(Y)inFovea"]
df_natural_pre_even.loc[df_natural_pre_even['FoveaEven'] >= 1,'FoveaEven'] = 1

df_natural_pre_even

# %%
df_natural_pre_even.loc[df_natural_pre_even['FoveaEven'] >= 1,'looking'] = 'look'
df_natural_pre_even.loc[df_natural_pre_even['FoveaEven'] <=0,'looking'] = 'not look'
df_natural_pre_even

# %%
df_natural_pre_odd['look_each_other'] = np.where(df_natural_pre_odd['FoveaOdd'] == df_natural_pre_even['FoveaEven'], 'Match', 'No Match') 
#create new column in df1 to check if  match
df_natural_pre_odd

# %%
df_natural_pre_even['look_each_other'] = np.where(df_natural_pre_even['FoveaEven'] == df_natural_pre_odd['FoveaOdd'], 'Match', 'No Match') 
#create new column in df1 to check if  match
df_natural_pre_even

# %%
df_natural_pre_odd.loc[df_natural_pre_odd.FoveaOdd <= 0, "look_each_other"] = "0: No Match"
df_natural_pre_odd

# %%
df_natural_pre_even.loc[df_natural_pre_even.FoveaEven <= 0, "look_each_other"] = "0: No Match"
df_natural_pre_even

# %%
df_natural_pre_odd

# %%
df_natural_pre_even

# %%
df_natural_pre_odd['look_each_other'].head(120).value_counts()

# %%
#Threshold = 13
#for index in df_natural_pre_odd.index:
#    print(df_natural_pre_odd['FoveaOdd'][index])
#def sanjit_algorithm(df_natural_pre_odd, threshold=13, step_size=125, new_column_name="percent"):
#    df_natural_pre_odd[new_column_name] = np.nan
#    for i in range(0, len(df_natural_pre_odd), step_size):
#        condition = (df_natural_pre_odd.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_natural_pre_odd.iloc[i][new_column_name] = 1 if condition else 0
#    return df_natural_pre_odd

#df_natural_pre_odd = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_natural_pre_odd = sanjit_algorithm(df_natural_pre_odd)
#df_natural_pre_odd

threshold = 13
current_rows = 0 

while current_rows < len(df_natural_pre_odd) + 125:
    selection = df_natural_pre_odd.loc[current_rows: current_rows + 125, 'FoveaOdd']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_natural_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_natural_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_natural_pre_odd.loc[current_rows + 125: len(df_natural_pre_odd), 'PercentOdd'] = 0

# %%
#Threshold = 13
#for index in df_natural_pre_even.index:
#    print(df_natural_pre_even['FoveaEven'][index])
#def sanjit_algorithm(df_natural_pre_even, threshold=13, step_size=125, new_column_name="percent"):
#    df_natural_pre_even[new_column_name] = np.nan
#    for i in range(0, len(df_natural_pre_even), step_size):
#        condition = (df_natural_pre_even.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_natural_pre_even.iloc[i][new_column_name] = 1 if condition else 0
#    return df_natural_pre_even

#df_natural_pre_even = pd.DataFrame(index=[random.randint(0, 1) for _ in range(125)])  
#df_natural_pre_even = sanjit_algorithm(df_natural_pre_even)
#df_natural_pre_even
threshold = 13
current_rows = 0 

while current_rows < len(df_natural_pre_even) + 125:
    selection = df_natural_pre_even.loc[current_rows: current_rows + 125, 'FoveaEven']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_natural_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_natural_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
        
df_natural_pre_even.loc[current_rows + 125: len(df_natural_pre_even), 'PercentEven'] = 0

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
natural_pre_files = glob.glob(path + "/*natural_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
natural_files_pre_odd = []
natural_files_pre_even = []

for file in natural_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        natural_files_pre_odd.append(file)
    else:
        natural_files_pre_even.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(natural_files_pre_odd):
    natural_files_pre_odd = pd.read_csv(filename, index_col=None, header=0)
    sec = natural_files_pre_odd.shape[0] / 125
    print(f"Natural_pre_odd, pair : {idx}, Total Rows : {natural_files_pre_odd.shape[0]}, seconds : {sec}")

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
natural_pre_files = glob.glob(path + "/*natural_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
natural_files_pre_odd = []
natural_files_pre_even = []

for file in natural_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 == 0:
        natural_files_pre_even.append(file)
    else:
        natural_files_pre_odd.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(natural_files_pre_even):
    natural_files_pre_even = pd.read_csv(filename, index_col=None, header=0)
    sec = natural_files_pre_even.shape[0] / 125
    print(f"Natural_pre_even, pair : {idx}, Total Rows : {natural_files_pre_even.shape[0]}, seconds : {sec}")

# %%
col = df_natural_pre_odd.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [117, 112, 115, 101, 104, 114, 110, 112, 119, 112, 118, 111, 120, 120, 109]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 117, 229, 344, 445, 549, 663, 773, 885, 1004, 1116, 1234, 1345, 1465, 1585]
rows_each_pair_end =   [117, 229, 344, 445, 549, 663, 773, 885, 1004, 1116, 1234, 1345, 1465, 1585, 1694]

# Create new column for percentage looking and not looking
df_natural_pre_odd["look_percentage"]=""
df_natural_pre_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_natural_pre_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_natural_pre_odd.loc[idx, ["look_percentage"]] = one_percentage
    df_natural_pre_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_natural_pre_odd

# %%
col = df_natural_pre_even.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [109, 112, 115, 101, 104, 117, 114, 110, 112, 119, 112, 118, 111, 120, 120]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 109, 221, 336, 437, 541, 658, 772, 882, 994, 1113, 1225, 1343, 1454, 1574]
rows_each_pair_end =   [109, 221, 336, 437, 541, 658, 772, 882, 994, 1113, 1225, 1343, 1454, 1574, 1694]

# Create new column for percentage looking and not looking
df_natural_pre_even["look_percentage"]=""
df_natural_pre_even["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_natural_pre_even.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_natural_pre_even.loc[idx, ["look_percentage"]] = one_percentage
    df_natural_pre_even.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_natural_pre_even

# %%
df_natural_pre_odd['ThresholdPercentage'] = np.where((df_natural_pre_odd['PercentOdd'] == 1) & (df_natural_pre_even['PercentEven'] == 1), '1', '0')
print (df_natural_pre_odd.iloc[:1875])

# %%
df_natural_pre_odd['ThresholdPercentage'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 

# %%


# %%
#df_natural_pre_odd_v2 = df_natural_pre_odd[['FoveaOdd']]
#df_natural_pre_odd_v2

# %%
#df_natural_pre_even_v2 = df_natural_pre_even[['FoveaEven']]
#df_natural_pre_even_v2

# %%
#df_even_odd_join5 = df_natural_pre_even_v2.merge(df_natural_pre_odd_v2, left_index=True, right_index=True, how='inner')
#df_even_odd_join5['look_match'] = [ x & y for x,y in zip(df_even_odd_join5['FoveaOdd'], df_even_odd_join5['FoveaEven'])]
#df_even_odd_join5['look_match'].value_counts()

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new/'
natural_post_files = glob.glob(path + "/*natural_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
natural_files_post_odd = []
natural_files_post_even = []

for file in natural_post_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        natural_files_post_odd.append(file)
    else:
        natural_files_post_even.append(file)

li_natural_post_odd = []
li_natural_post_even = []

# ###############################################
# Combine all averted pre odd files
for filename in natural_files_post_odd:
    df_natural_post_odd = pd.read_csv(filename, index_col=None, header=0)
    df_natural_post_odd.append(df_natural_post_odd)
# Populate all dataframes into one dataframe
df_natural_post_odd = pd.concat(li_averted_pre_odd, axis=0, ignore_index=True)
# Remove row where there is NaN value
df_natural_post_odd = df_natural_post_odd.dropna()
df_natural_post_odd = df_natural_post_odd.reset_index(drop=True)
# Remove space before column names
df_natural_post_odd_new_columns = df_natural_post_odd.columns.str.replace(
    ' ', '')
df_natural_post_odd.columns = df_natural_post_odd_new_columns
# df_averted_pre_odd.head()

# ###############################################
# Combine all averted pre even files
for filename in natural_files_post_even:
    df_natural_post_even = pd.read_csv(filename, index_col=None, header=0)
    li_natural_post_even.append(df_natural_post_even)

# Populate all dataframes into one dataframe
df_natural_post_even = pd.concat(li_natural_post_even, axis=0, ignore_index=True)

# Remove row where there is NaN value
df_natural_post_even = df_natural_post_even.dropna()
df_natural_post_even = df_natural_post_even.reset_index(drop=True)

# Remove space before column names
df_natural_post_even_new_columns = df_natural_post_even.columns.str.replace(
    ' ', '')
df_natural_post_even.columns = df_natural_post_even_new_columns
# df_averted_pre_even.head()

# %%
# Gaze direction (right eye)
df_natural_post_odd['GazeDirectionRight(X)Degree'] = df_natural_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_post_odd['GazeDirectionRight(X)Degree'] = df_natural_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_post_odd['GazeDirectionRight(Y)Degree'] = df_natural_post_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_natural_post_odd['GazeDirectionLeft(X)Degree'] = df_natural_post_odd.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_natural_post_odd['GazeDirectionLeft(Y)Degree'] = df_natural_post_odd.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_natural_post_odd['GazeDirectionRight(X)inFovea'] = df_natural_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_natural_post_odd['GazeDirectionRight(Y)inFovea'] = df_natural_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_natural_post_odd['GazeDirectionLeft(X)inFovea'] = df_natural_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_natural_post_odd['GazeDirectionLeft(Y)inFovea'] = df_natural_post_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_natural_post_odd.head(14874)

# %%
df_natural_post_odd["FoveaOdd"] = df_natural_post_odd["GazeDirectionRight(X)inFovea"] + df_natural_post_odd["GazeDirectionLeft(X)inFovea"] + df_natural_post_odd["GazeDirectionRight(Y)inFovea"] + df_natural_post_odd["GazeDirectionLeft(Y)inFovea"]
df_natural_post_odd.loc[df_natural_post_odd['FoveaOdd'] >= 1,'FoveaOdd'] = 1

df_natural_post_odd

# %%
df_natural_post_odd.loc[df_natural_post_odd["FoveaOdd"] >= 1,'looking'] = 'look'
df_natural_post_odd.loc[df_natural_post_odd["FoveaOdd"] <=0,'looking'] = 'not look'
df_natural_post_odd

# %%
# Gaze direction (right eye)
df_natural_post_even['GazeDirectionRight(X)Degree'] = df_natural_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_post_even['GazeDirectionRight(X)Degree'] = df_natural_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionRight(X)'], x['GazeDirectionRight(Y)']), axis=1)
df_natural_post_even['GazeDirectionRight(Y)Degree'] = df_natural_post_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionRight(Y)'], x['GazeDirectionRight(Z)']), axis=1)

# Gaze direction (left eye)
df_natural_post_even['GazeDirectionLeft(X)Degree'] = df_natural_post_even.apply(lambda x: gaze_direction_in_x_axis_degree(x['GazeDirectionLeft(X)'], x['GazeDirectionLeft(Y)']), axis=1)
df_natural_post_even['GazeDirectionLeft(Y)Degree'] = df_natural_post_even.apply(lambda x: gaze_direction_in_y_axis_degree(x['GazeDirectionLeft(Y)'], x['GazeDirectionLeft(Z)']), axis=1)
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_natural_post_even['GazeDirectionRight(X)inFovea'] = df_natural_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_natural_post_even['GazeDirectionRight(Y)inFovea'] = df_natural_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_natural_post_even['GazeDirectionLeft(X)inFovea'] = df_natural_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_natural_post_even['GazeDirectionLeft(Y)inFovea'] = df_natural_post_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_natural_post_even["FoveaEven"] = df_natural_post_even["GazeDirectionRight(X)inFovea"] + df_natural_post_even["GazeDirectionLeft(X)inFovea"] + df_natural_post_even["GazeDirectionRight(Y)inFovea"] + df_natural_post_even["GazeDirectionLeft(Y)inFovea"]
df_natural_post_even.loc[df_natural_post_even['FoveaEven'] >= 1,'FoveaEven'] = 1

df_natural_post_even

# %%
df_natural_post_even.loc[df_natural_post_even['FoveaEven'] >= 1,'looking'] = 'look'
df_natural_post_even.loc[df_natural_post_even['FoveaEven'] <=0,'looking'] = 'not look'
df_natural_post_even

# %%
#df_natural_post_odd['look_each_other'] = np.where(df_natural_post_odd['FoveaOdd'] == df_natural_post_even['FoveaEven'], '1', '0') 
#df_natural_post_odd

# %%
#df_natural_post_even['look_each_other'] = np.where( df_natural_post_even['FoveaEven'] == df_natural_post_odd['FoveaOdd'], '1', '0') 
#df_natural_post_even

# %%
df_natural_post_odd.loc[df_natural_post_odd.FoveaOdd <= 0, "look_each_other"] = "0"
df_natural_post_odd

# %%
df_natural_post_even.loc[df_natural_post_even.FoveaEven <= 0, "look_each_other"] = "0"
df_natural_post_even

# %%
df_natural_post_odd

# %%
df_natural_post_even

# %%
#df_natural_post_even_v2 = df_natural_post_even[['FoveaEven']]
#df_natural_post_even_v2

# %%
#df_even_odd_join6 = df_natural_post_even_v2.merge(df_natural_pre_odd_v2, left_index=True, right_index=True, how='inner')
#df_even_odd_join6['look_match'] = [ x & y for x,y in zip(df_even_odd_join6['FoveaOdd'], df_even_odd_join6['FoveaEven'])]
#df_even_odd_join6['look_match'].value_counts()

# %%
df_natural_post_odd['look_each_other'].value_counts(120)

# %%
df_natural_post_odd['look_each_other'].iloc[:60]

# %%
threshold = 13
current_rows = 0 

while current_rows < len(df_natural_post_odd) + 125:
    selection = df_natural_post_odd.loc[current_rows: current_rows + 125, 'FoveaOdd']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_natural_post_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_natural_post_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_natural_post_odd.loc[current_rows + 125: len(df_natural_post_odd), 'PercentOdd'] = 0

# %%
threshold = 13
current_rows = 0 

while current_rows < len(df_natural_post_even) + 125:
    selection = df_natural_post_even.loc[current_rows: current_rows + 125, 'FoveaEven']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_natural_post_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_natural_post_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
        
df_natural_post_even.loc[current_rows + 125: len(df_natural_post_even), 'PercentEven'] = 0

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
natural_post_files = glob.glob(path + "/*natural_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
natural_files_post_odd = []
natural_files_post_even = []

for file in natural_post_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        natural_files_post_odd.append(file)
    else:
        natural_files_post_even.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(natural_files_post_odd):
    df_natural_post_odd = pd.read_csv(filename, index_col=None, header=0)
    sec = df_natural_post_odd.shape[0] / 125
    print(f"Natural_post_odd, pair : {idx}, Total Rows : {df_natural_post_odd.shape[0]}, seconds : {sec}")

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
natural_post_files = glob.glob(path + "/*natural_post*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
natural_files_post_odd = []
natural_files_post_even = []

for file in natural_post_files:
    if int(re.search(pattern, file).group(1)) % 2 == 0:
        natural_files_post_even.append(file)
    else:
        natural_files_post_odd.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(natural_files_post_even):
    df_natural_post_even = pd.read_csv(filename, index_col=None, header=0)
    sec = df_natural_post_even.shape[0] / 125
    print(f"Natural_post_even, pair : {idx}, Total Rows : {df_natural_post_even.shape[0]}, seconds : {sec}")

# %%
col = df_natural_post_odd.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [98, 119, 102, 101, 104, 117, 105, 119, 116, 117, 116, 101, 91, 120, 91]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 98, 217, 319, 420, 524, 641, 746, 865, 981, 1098, 1214, 1315, 1406, 1526]
rows_each_pair_end =   [98, 217, 319, 420, 524, 641, 746, 865, 981, 1098, 1214, 1315, 1406, 1526, 1617]

# Create new column for percentage looking and not looking
df_natural_post_odd["look_percentage"]=""
df_natural_post_odd["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_natural_post_odd.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_natural_post_odd.loc[idx, ["look_percentage"]] = one_percentage
    df_natural_post_odd.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_natural_post_odd

# %%
col = df_natural_post_even.loc("look_each_other")
# Put the seconds of each pair REPLACE THESE VALUES
# The number of elements has to be 15 !!!!
seconds = [91, 119, 102, 101, 104, 98, 117, 105, 119, 116, 117, 116, 101, 91, 120]
# Index to extract data based on the above seconds
# IMPORTANT!! INDEX. Don't get confused. Please do simple addition !!
rows_each_pair_begin = [0, 91, 210, 312, 413, 517, 615, 732, 837, 956, 1072, 1189, 1305, 1406, 1497]
rows_each_pair_end =   [91, 210, 312, 413, 517, 615, 732, 837, 965, 1072, 1189, 1305, 1406, 1497, 1617]

# Create new column for percentage looking and not looking
df_natural_post_even["look_percentage"]=""
df_natural_post_even["not_look_percentage"]=""

for idx, value in enumerate(rows_each_pair_begin):
    df_temp = df_natural_post_even.get_loc[np.r_[rows_each_pair_begin[idx]:rows_each_pair_end[idx]], :]
    one_counter = df_temp.loc[df_temp.look_each_other == 1, "look_each_other" ].count()
    zero_counter = df_temp.loc[df_temp.look_each_other == 0, "look_each_other" ].count()
    # Calculate average
    one_percentage = (one_counter / 125) * 100
    zero_percentage = 100 - one_percentage
    # Assign percentage on new column
    df_natural_post_even.loc[idx, ["look_percentage"]] = one_percentage
    df_natural_post_even.loc[idx, ["not_look_percentage"]] = zero_percentage
    # ic(idx, one_counter, zero_counter, one_percentage, zero_percentage)

df_natural_post_even

# %%
df_natural_post_odd['ThresholdPercentage'] = np.where((df_natural_post_odd['PercentOdd'] == 1) & (df_natural_post_even['PercentEven'] == 1), '1', '0')
print (df_natural_post_odd.iloc[:1875])

# %%
df_natural_post_odd['ThresholdPercentage'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

# %%
