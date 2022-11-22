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


# %% [markdown]
# ## Function to populate odd and even subjects into one separate dataFrame (averted_pre) 

# %%
def combine_eye_data_into_dataframe(path2files: str, tag:str):
    
    """
        Combine all cleaned eye tracker data, eg. averted_pre, into one dataframe.
        It also involves pre-processing (replacing missing values with average value of columns
        where they are). However, the return value is dataframe of odd and even subject that is separated

    Args:
        path2files (str): Path to a directory where all cleaned eye tracker files are stored
        tag (str): eye gaze condition, ie. averted_pre, averted_post, direct_pre, direct_post, natural_pre, natural_post

    Returns:
        tuple: Consists of two dataframes :
               1. ODD subject [index = 0]
               2. and of EVEN [index = 1]
    """

    gaze_keyword= "/*" + tag + "*.csv"
    pre_files = glob.glob(path2files + gaze_keyword)
    pattern = re.compile(r"[S]+(\d+)\-")
    files_pre_odd = []
    files_pre_even = []

    for file in pre_files:
        if int(re.search(pattern, file).group(1)) % 2 != 0:
            files_pre_odd.append(file)
        else:
            files_pre_even.append(file)

    li_pre_odd = []
    li_pre_even = []

    # ###############################################
    # Combine all pre odd files
    for filename in files_pre_odd:
        df_pre_odd = pd.read_csv(filename, index_col=None, header=0)
        li_pre_odd.append(df_pre_odd)
    # Populate all dataframes into one dataframe
    df_pre_odd = pd.concat(li_pre_odd, axis=0, ignore_index=True)

    # Replace missing values with averages of columns where they are
    df_pre_odd.fillna(df_pre_odd.mean(), inplace=True)

    df_pre_odd = df_pre_odd.reset_index(drop=True)
    # Remove space before column names
    df_pre_odd_new_columns = df_pre_odd.columns.str.replace(
        ' ', '')
    df_pre_odd.columns = df_pre_odd_new_columns

    # ###############################################
    # Combine all pre even files
    for filename in files_pre_even:
        df_pre_even = pd.read_csv(filename, index_col=None, header=0)
        li_pre_even.append(df_pre_even)

    # Populate all dataframes into one dataframe
    df_pre_even = pd.concat(li_pre_even, axis=0, ignore_index=True)


    # Replace missing values with averages of columns where they are
    df_pre_even.fillna(df_pre_even.mean(), inplace=True)

    df_pre_even = df_pre_even.reset_index(drop=True)

    # Remove space before column names
    df_pre_even_new_columns = df_pre_even.columns.str.replace(
        ' ', '')
    df_pre_even.columns = df_pre_even_new_columns
    
    return df_pre_odd, df_pre_even

# %% [markdown]
# ### Running a function of combine_eye_data_into_dataframe

# %%
tag = "averted_pre"
path2files = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EyeTracker/raw_experimental_eye_data/raw_combined_experimental_eye_data/raw_cleaned_combined_experimental_eye_data"

df_averted = combine_eye_data_into_dataframe(path2files, tag)
print("averted_pre ODD subjects")
df_averted_pre_odd = df_averted[0]
# df_averted_pre_odd.head()

print("averted_pre EVEN subjects")
df_averted_pre_even = df_averted[1]



# %% [markdown]
# ## Cut off the unneccessary columns of averted_pre EVEN subjects

# %%
# NOTE: No idea, why for some reasons the number of columns for df_averted_pre_even is almost double then df_averted_pre_odd dataframe
# So we need to cut it off

# This is just to check the number of columns of odd subject
# odd_cols =  [x for x in df_averted_pre_odd.columns]
# print(len(odd_cols))

# # This is just to check the number of columns of even subject
# even_cols =  [x for x in df_averted_pre_even.columns]
# print(len(even_cols))

df_averted_pre_even = df_averted_pre_even.iloc[:,0:24]



# %% [markdown]
# ## Function to convert cartesian to degree
# Cartesian is a default value that is resulted from HTC Vive pro

# %%
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
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_averted_pre_odd['GazeDirectionRight(X)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_averted_pre_odd['GazeDirectionRight(Y)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_averted_pre_odd['GazeDirectionLeft(X)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_averted_pre_odd['GazeDirectionLeft(Y)inFovea'] = df_averted_pre_odd.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

#df_averted_pre_odd.head(14874)

#df_averted_pre_odd.iloc[2, 'GazeDirectionRight(X)Degree'] 


# %%
df_averted_pre_odd.head()

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
# %% Give mark 1 for GazeDirection Y/X Degree that falls under fovea area (30 degrees), otherwise 0

df_averted_pre_even['GazeDirectionRight(X)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(X)Degree']), axis=1)
df_averted_pre_even['GazeDirectionRight(Y)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionRight(Y)Degree']), axis=1)
df_averted_pre_even['GazeDirectionLeft(X)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(X)Degree']), axis=1)
df_averted_pre_even['GazeDirectionLeft(Y)inFovea'] = df_averted_pre_even.apply(lambda x: check_degree_within_fovea(x['GazeDirectionLeft(Y)Degree']), axis=1)

# %%
df_averted_pre_odd.head()

# %%
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionRight(X)inFovea'] <= 0,'FoveaOdd'] = 0
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionRight(X)inFovea'] >= 1,'FoveaOdd'] = 1 
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionLeft(X)inFovea']  <=0, 'FoveaOdd'] = 0
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionLeft(X)inFovea']  >=1, 'FoveaOdd'] = 1
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionRight(Y)inFovea'] <=0,'FoveaOdd'] = 0
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionRight(Y)inFovea'] >=1,'FoveaOdd'] = 1
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionLeft(Y)inFovea']  <=0, 'FoveaOdd'] = 0
#df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionLeft(Y)inFovea']  >=1, 'FoveaOdd'] = 1

#df_averted_pre_odd['FoveaOdd'] = df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionRight(X)inFovea'] + df_averted_pre_odd.loc[df_averted_pre_odd['GazeDirectionLeft(X)inFovea']

df_averted_pre_odd["FoveaOdd"] = df_averted_pre_odd["GazeDirectionRight(X)inFovea"] + df_averted_pre_odd["GazeDirectionLeft(X)inFovea"] + df_averted_pre_odd["GazeDirectionRight(Y)inFovea"] + df_averted_pre_odd["GazeDirectionLeft(Y)inFovea"]
df_averted_pre_odd.loc[df_averted_pre_odd['FoveaOdd'] >= 1,'FoveaOdd'] = 1

df_averted_pre_odd


# %%
df_averted_pre_odd.loc[df_averted_pre_odd['FoveaOdd'] >= 1,'looking'] = 'look'
df_averted_pre_odd.loc[df_averted_pre_odd['FoveaOdd'] <=0,'looking'] = 'not look'
df_averted_pre_odd

# %%
#df_averted_pre_even.loc[df_averted_pre_even['GazeDirectionRight(X)inFovea'] <= 0,'FoveaEven'] = 0
#df_averted_pre_even.loc[df_averted_pre_even['GazeDirectionLeft(X)inFovea']  <=0, 'FoveaEven'] = 0
#df_averted_pre_even.loc[df_averted_pre_even['GazeDirectionLeft(X)inFovea']  ==1, 'FoveaEven'] = 1
#df_averted_pre_even.loc[df_averted_pre_even['GazeDirectionRight(Y)inFovea'] <= 0,'FoveaEven'] = 0
#df_averted_pre_even.loc[df_averted_pre_even['GazeDirectionLeft(Y)inFovea']  ==1, 'FoveaEven'] = 1

df_averted_pre_even["FoveaEven"] = df_averted_pre_even["GazeDirectionRight(X)inFovea"] + df_averted_pre_even["GazeDirectionLeft(X)inFovea"] + df_averted_pre_even["GazeDirectionRight(Y)inFovea"] + df_averted_pre_even["GazeDirectionLeft(Y)inFovea"]

df_averted_pre_even.loc[df_averted_pre_even['FoveaEven'] >= 1,'FoveaEven'] = 1

df_averted_pre_even.head(10)
#df_averted_pre_even

# %%
df_averted_pre_even.loc[df_averted_pre_even['FoveaEven'] >= 1,'looking'] = 'look'
df_averted_pre_even.loc[df_averted_pre_even['FoveaEven'] <=0,'looking'] = 'not look'
df_averted_pre_even

# %%
#df_averted_pre_odd['FoveaOdd'].value_counts()
#df_averted_pre_odd = pd.DataFrame({'a':list('abssbab')})
#df_averted_pre_odd.groupby('FoveaOdd').count()
# construct the new column
#looking_each_other = df_averted_pre_even['FoveaEven'].copy().astype('float32')

# run the sliding window
#for t in range(len(df_averted_pre_odd.index) - 1800):
#    odd_col = df_averted_pre_odd['FoveaOdd'][t:t+1800].copy()
#    even_col = df_averted_pre_even['FoveaEven'][t:t+1800].copy()
#    sum_col = odd_col + even_col
#    num_of_matches = (sum_col==2).sum()
#    proportion_of_match = num_of_matches / 1800.0
#    print(proportion_of_match)
#    looking_each_other[t] = proportion_of_match

# add looking_each_other to the original table
df_averted_pre_odd['look_each_other'] = np.where(df_averted_pre_odd['FoveaOdd'] == df_averted_pre_even['FoveaEven'], '1', '0') 
#create new column in df1 to check if prices match
df_averted_pre_odd

# %%
df_averted_pre_even['look_each_other'] = np.where(df_averted_pre_even['FoveaEven'] == df_averted_pre_odd['FoveaOdd'], '1', '0') 

df_averted_pre_even

# %%
df_averted_pre_odd.loc[df_averted_pre_odd.FoveaOdd <= 0, "look_each_other"] = "0"
df_averted_pre_odd

# %%
df_averted_pre_even.loc[df_averted_pre_even.FoveaEven <= 0, "look_each_other"] = "0"
df_averted_pre_even

# %%
df_averted_pre_odd['look_each_other'].head(50)

# %%
df_averted_pre_even['look_each_other'].head(50)

# %%
df_averted_pre_odd['look_each_other'].head(120).value_counts()

# %%
#df_averted_pre_odd['look_each_other'].value_counts()
df_averted_pre_odd_v2 = df_averted_pre_odd[['FoveaOdd']]
df_averted_pre_odd_v2

# %%
#df_averted_pre_even['look_each_other'].value_counts()
df_averted_pre_even_v2 = df_averted_pre_even[['FoveaEven']]
df_averted_pre_even_v2

# %%
df_even_odd_join = df_averted_pre_even_v2.merge(df_averted_pre_odd_v2, left_index=True, right_index=True, how='inner')
df_even_odd_join['look_match'] = [ x & y for x,y in zip(df_even_odd_join['FoveaOdd'], df_even_odd_join['FoveaEven'])]
df_even_odd_join['look_match'].value_counts()

# %%
df_even_odd_join

# %%
df_averted_pre_odd['look_each_other'].iloc[100:120]

# %%
#Threshold = 13
#for index in df_averted_pre_odd.index:
#    print(df_averted_pre_odd['FoveaOdd'][index])
#def sanjit_algorithm(df_averted_pre_odd, threshold=13, step_size=125, new_column_name="percent"):
#    df_averted_pre_odd[new_column_name] = np.nan
#    for i in range(0, len(df_averted_pre_odd), step_size):
#        condition = (df_averted_pre_odd.iloc[i:i+step_size].index == 1).sum() > threshold
#        df_averted_pre_odd.iloc[i][new_column_name] = 1 if condition else 0
#    return df_averted_pre_odd

#df_averted_pre_odd = pd.DataFrame(index=[random.randint(0, 1) for _ in range(208250)])  
#df_averted_pre_odd = sanjit_algorithm(df_averted_pre_odd)
#df_averted_pre_odd

threshold = 13
current_rows = 0 

while current_rows < len(df_averted_pre_odd) + 125:
    selection = df_averted_pre_odd.loc[current_rows: current_rows + 125, 'FoveaOdd']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_averted_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 1
    else:
        df_averted_pre_odd.loc[current_rows: current_rows + 125, 'PercentOdd'] = 0
        
    current_rows += 125
        
df_averted_pre_odd.loc[current_rows + 125: len(df_averted_pre_odd), 'PercentOdd'] = 0
df_averted_pre_odd.shape

# %%
threshold = 13
current_rows = 0 

while current_rows < len(df_averted_pre_even) + 125:
    selection = df_averted_pre_even.loc[current_rows: current_rows + 125, 'FoveaEven']
    
    only_ones = [num for num in list(selection) if num == 1]
    count_of_ones = len(only_ones)
    
    print(count_of_ones, current_rows)
    
    if count_of_ones >= 13:
        df_averted_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 1
    else:
        df_averted_pre_even.loc[current_rows: current_rows + 125, 'PercentEven'] = 0
        
    current_rows += 125
    

df_averted_pre_even.loc[current_rows + 125: len(df_averted_pre_even), 'PercentEven'] = 0



# %%
df_averted_pre_even

# %%
df_averted_pre_odd['ThresholdPercentage'] = np.where((df_averted_pre_odd['PercentOdd'] == 1) & (df_averted_pre_even['PercentEven'] == 1), '1', '0')
print (df_averted_pre_odd.iloc[:1875])

# %%
df_averted_pre_odd['ThresholdPercentage'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 

# %%
df_averted_pre_odd['ThresholdPercentage']

# %%
# use your path
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
averted_pre_files = glob.glob(path + "/*averted_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
averted_files_pre_odd = []
averted_files_pre_even = []

for file in averted_pre_files:
    if int(re.search(pattern, file).group(1)) % 2 != 0:
        averted_files_pre_odd.append(file)
    else:
        averted_files_pre_even.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(averted_files_pre_odd):
    df_averted_pre_odd = pd.read_csv(filename, index_col=None, header=0)
    sec = df_averted_pre_odd.shape[0] / 125
    print(f"Averted_pre_odd, pair : {idx}, Total Rows : {df_averted_pre_odd.shape[0]}, seconds : {sec}")

# %%
path = r'C:/Users/sanji/Downloads/eye_tracker_data_clean_new'
averted_pre_files = glob.glob(path + "/*averted_pre*.csv")
pattern = re.compile(r"[S]+(\d+)\-")
averted_files_pre_odd = []
averted_files_pre_even = []

for file in averted_pre_files:
    if int(re.search(pattern, file).group(1)) % 2== 0:
        averted_files_pre_even.append(file)
    else:
        averted_files_pre_odd.append(file)

# Get how many seconds for each pair
for idx, filename in enumerate(averted_files_pre_even):
    df_averted_pre_even = pd.read_csv(filename, index_col=None, header=0)
    sec = df_averted_pre_even.shape[0] / 125
    print(f"Averted_pre_even, pair : {idx}, Total Rows : {df_averted_pre_even.shape[0]}, seconds : {sec}")

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
