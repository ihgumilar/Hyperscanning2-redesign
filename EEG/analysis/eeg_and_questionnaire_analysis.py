# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,cell_depth,-all
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
import os 
import re
import pathlib
import pandas as pd
import numpy as np
import pingouin as pg
from pandas.io.api import read_pickle
from collections import namedtuple
from scipy.special import logsumexp
from statistics import mean


# %% [markdown]
# ## Count significant connections for each eye condition: 
# ### Divided into 4 frequencies (theta, alpha, beta, and gamma)

# %%
def total_significant_connections(path: str):

    """Count a number of significant connections for a certain eye condition, eg. averted_pre.
       Divided into different algorithms (ccorr, coh, and plv) and frequencies (theta, alpha, beta, and gamma)

    Parameters :
        path (str) : A folder that contains *pkl file which contains actual scores of connections.
                     Each *.pkl file will have a lenght of 4 (the order is theta, alpha, beta, and gamma)
        
    Returns:
        all_connections (namedtuple): it returns multiple values. The order is described below:

        total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
        total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
        total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections,

    """
    
    results = namedtuple("results",
    ["total_sig_ccorr_theta_connections", "total_sig_ccorr_alpha_connections", "total_sig_ccorr_beta_connections", "total_sig_ccorr_gamma_connections",
    "total_sig_coh_theta_connections", "total_sig_coh_alpha_connections", "total_sig_coh_beta_connections", "total_sig_coh_gamma_connections",
    "total_sig_plv_theta_connections", "total_sig_plv_alpha_connections", "total_sig_plv_beta_connections", "total_sig_plv_gamma_connections"])

    files = os.listdir(path)
    # Create new list to count the number of significant connection (eg. list_at, list_aa, list_ab, list_ag)
    ccorr_sig_connections = []
    coh_sig_connections = []
    plv_sig_connections = []

    # Separate files into different container according to algorithm
    for file in files:
        # ccorr
        if ("actual_score_data" in file and "ccorr" in file):
            ccorr_sig_connections.append(file)
            # Sort the list
            ccorr_sig_connections.sort()
        # coh
        elif ("actual_score_data" in file and "coh" in file) :
            coh_sig_connections.append(file)
            # Sort the list
            coh_sig_connections.sort()
        # plv
        elif ("actual_score_data" in file and "plv" in file) :
            plv_sig_connections.append(file)
            # Sort the list
            plv_sig_connections.sort()

    # Define list for ccorr per frequency
    total_sig_ccorr_theta_connections = []
    total_sig_ccorr_alpha_connections = []
    total_sig_ccorr_beta_connections = []
    total_sig_ccorr_gamma_connections = []

    # Define list for coh per frequency
    total_sig_coh_theta_connections = []
    total_sig_coh_alpha_connections = []
    total_sig_coh_beta_connections = []
    total_sig_coh_gamma_connections = []

    # Define list for plv per frequency
    total_sig_plv_theta_connections = []
    total_sig_plv_alpha_connections = []
    total_sig_plv_beta_connections = []
    total_sig_plv_gamma_connections = []


    # Count significant connection for ccorr algorithm and separate into 4 frequencies:
    # theta, alpha, beta, and gamma
    for file in ccorr_sig_connections:
        ccorr_file_2_read = os.path.join(path, file)
        ccorr_file = read_pickle(ccorr_file_2_read)
        
        # Theta = 0th index in the list
        sig_ccorr_theta_connections = len(ccorr_file[0])
        total_sig_ccorr_theta_connections.append(sig_ccorr_theta_connections)

        # Alpha = 1st index in the list
        sig_ccorr_alpha_connections = len(ccorr_file[1])
        total_sig_ccorr_alpha_connections.append(sig_ccorr_alpha_connections)

        # Beta = 2nd index in the list
        sig_ccorr_beta_connections = len(ccorr_file[2])
        total_sig_ccorr_beta_connections.append(sig_ccorr_beta_connections)

        # Gamma = 3rd index in the list
        sig_ccorr_gamma_connections = len(ccorr_file[3])
        total_sig_ccorr_gamma_connections.append(sig_ccorr_gamma_connections)


    # Count significant connection for coh algorithm and separate into 4 frequencies:
    # theta, alpha, beta, and gamma
    for file in coh_sig_connections:
        coh_file_2_read = os.path.join(path, file)
        coh_file = read_pickle(coh_file_2_read)
        
        # Theta = 0th index in the list
        sig_coh_theta_connections = len(coh_file[0])
        total_sig_coh_theta_connections.append(sig_coh_theta_connections)

        # Alpha = 1st index in the list
        sig_coh_alpha_connections = len(coh_file[1])
        total_sig_coh_alpha_connections.append(sig_coh_alpha_connections)

        # Beta = 2nd index in the list
        sig_coh_beta_connections = len(coh_file[2])
        total_sig_coh_beta_connections.append(sig_coh_beta_connections)

        # Gamma = 3rd index in the list
        sig_coh_gamma_connections = len(coh_file[3])
        total_sig_coh_gamma_connections.append(sig_coh_gamma_connections)


    # Count significant connection for plv algorithm and separate into 4 frequencies:
    # theta, alpha, beta, and gamma
    for file in plv_sig_connections:
        plv_file_2_read = os.path.join(path, file)
        plv_file = read_pickle(plv_file_2_read)
        
        # Theta = 0th index in the list
        sig_plv_theta_connections = len(plv_file[0])
        total_sig_plv_theta_connections.append(sig_plv_theta_connections)

        # Alpha = 1st index in the list
        sig_plv_alpha_connections = len(plv_file[1])
        total_sig_plv_alpha_connections.append(sig_plv_alpha_connections)

        # Beta = 2nd index in the list
        sig_plv_beta_connections = len(plv_file[2])
        total_sig_plv_beta_connections.append(sig_plv_beta_connections)

        # Gamma = 3rd index in the list
        sig_plv_gamma_connections = len(plv_file[3])
        total_sig_plv_gamma_connections.append(sig_plv_gamma_connections)

    all_connections = results(total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
    total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
    total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections)
    
    return all_connections

# %% [markdown]
# ## Running function to count_significant_connections

# %%
path_dir_averted_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre/"
path_dir_averted_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_post/"
path_dir_direct_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_pre/"
path_dir_direct_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_post/"
path_dir_natural_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_pre/"
path_dir_natural_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_post/"

averted_post = total_significant_connections(path_dir_averted_post)
averted_pre = total_significant_connections(path_dir_averted_pre)
direct_post = total_significant_connections(path_dir_direct_post)
direct_pre = total_significant_connections(path_dir_direct_pre)
natural_post = total_significant_connections(path_dir_natural_post)
natural_pre = total_significant_connections(path_dir_natural_pre)

print(averted_post[9])
print(averted_pre[9])
print("")
print(direct_post[9])
print(direct_pre[9])
print("")
print(natural_post[9])
print(natural_pre[9])

# %% [markdown]
# ## 1.1. Find difference of number of connections between pre and post (Averted)
# NOTE : The variable name is exactly the same with section 2.1 . Make sure you run the function of total_significant_connections first !!. 
#
# IMPORTANT : Change no. 9 to whatever condition that you want to test. See the multiple output of total_significant_connections function

# %%
# Difference between PLV theta connections (post - pre) - averted eye condition

diff_averted_plv_theta = [x -y for x,y in zip(averted_post[9], averted_pre[9])]
print(F"Difference averted plv theta : {diff_averted_plv_theta}")

# Difference between PLV theta connections (post - pre) - direct eye condition

diff_direct_plv_theta = [x -y for x,y in zip(direct_post[9], direct_pre[9])]
print(F"Difference direct plv theta : {diff_direct_plv_theta}")

# Difference between PLV theta connections (post - pre) - natural eye condition

diff_natural_plv_theta = [x -y for x,y in zip(natural_post[9], natural_pre[9])]
print(F"Difference natural plv theta : {diff_natural_plv_theta}")


# %% [markdown]
# ## 1.2. Calculate statistical difference using friedman's test (non-parametric)
# NOTE : The variable name is exactly the same with section 2.2. Make sure you run the function of total_significant_connection first !!

# %%

# Combine the above lists and turn them into dataframe
combine_plv_theta = []
combine_plv_theta.append(diff_averted_plv_theta)
combine_plv_theta.append(diff_direct_plv_theta)
combine_plv_theta.append(diff_natural_plv_theta)

df_averted_plv_theta = pd.DataFrame(combine_plv_theta).transpose()
df_averted_plv_theta.columns = ["averted_plv_theta", "direct_plv_theta", "natural_plv_theta"]

print("PLV theta (averted vs direct vs natural)")
pg.friedman(df_averted_plv_theta)
   
# df

# %% [markdown]
# ### Calculate significant differences for all eye conditions, algorithms, and frequency - Still error
# Because the value of ccorr connections is zero

# %%
# for i in range(len(averted_pre)):

#     # Difference between PLV connections (post - pre) - averted eye condition
#     diff_averted = [x -y for x,y in zip(averted_post[i], averted_pre[i])]

#     # Difference between PLV connections (post - pre) - direct eye condition
#     diff_direct = [x -y for x,y in zip(direct_post[i], direct_pre[i])]

#     # Difference between PLV connections (post - pre) - natural eye condition
#     diff_natural = [x -y for x,y in zip(natural_post[i], natural_pre[i])]

#     # Combine the above lists and turn them into dataframe
#     combine_data = []
#     combine_data.append(diff_averted)
#     combine_data.append(diff_direct)
#     combine_data.append(diff_natural)

#     df_combine = pd.DataFrame(combine_data).transpose()
#     df_combine.columns = ["averted", "direct", "natural"]

#     print(F"Condition - {i}")
#     pg.friedman(combine_data)
#     print("")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Average of Actual score %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Loop for all significant connections
# e.g., Pre_ccorr_combined_pair_S1_and_S2_actual_score_data.pkl (consists of 4 lists - theta, alpha, beta, & gamma)

# Set up to directory path of significant connection, averted_pre
# Gather all files that contain _connection_data keyword and put into a list (list_a)

# Create new list to count the number of significant connection (eg. list_at, list_aa, list_ab, list_ag)
# Loop list_a
# Get the first list (e.g.theta) for each subject
# put into another list (list_at)
# Get the second list (e.g.alpha) for each subject
# put into another list (list_aa)
# Get the third list (e.g.beta) for each subject
# put into another list (list_ab)
# Get the fourth list (e.g.gamma) for each subject
# put into another list (list_ag)

#  Use this code https://github.com/ihgumilar/Hyperscanning2-redesign/issues/32
# to count average significant actual score of specific connections out of all pairs (from dictionary), which have key
# Apply that code to list_at, list_aa, list_ab, list_ag for each eye condition, eg. averted_pre
# (NOTE: It seems that the code is not working yet when we combine actual score from all subjects)

# Repeat the same procedure for all other eye conditions,eg. averted_post, etc


# %%
list_a = pd.read_pickle(
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre/Pre_plv_combined_pair_S1_and_S2_actual_score_data.pkl"
)

list_temp = list_a[0]
# list_temp.append(list_a[1])
# list_temp.append(list_a[2])
# print(list_a[0])

# %%
a = [0,0,0,0]
np.mean(a)


# %% [markdown]
# ## Function to calculate average actual score

# %%

def average_actual_score(path: str):

    """Count a number of significant connections for a certain eye condition, eg. averted_pre.
       Divided into different algorithms (ccorr, coh, and plv) and frequencies (theta, alpha, beta, and gamma)

    Parameters :
        path (str) : A folder that contains *pkl file which contains actual scores of connections.
                     Each *.pkl file will have a lenght of 4 (the order is theta, alpha, beta, and gamma)
        
    Returns:
        all_connections (namedtuple): it returns multiple values. The order is described below:

        total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
        total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
        total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections,

    """
    
    results = namedtuple("results",
    ["total_sig_ccorr_theta_connections", "total_sig_ccorr_alpha_connections", "total_sig_ccorr_beta_connections", "total_sig_ccorr_gamma_connections",
    "total_sig_coh_theta_connections", "total_sig_coh_alpha_connections", "total_sig_coh_beta_connections", "total_sig_coh_gamma_connections",
    "total_sig_plv_theta_connections", "total_sig_plv_alpha_connections", "total_sig_plv_beta_connections", "total_sig_plv_gamma_connections"])

    files = os.listdir(path)
    # Create new list to count the number of significant connection (eg. list_at, list_aa, list_ab, list_ag)
    ccorr_sig_connections = []
    coh_sig_connections = []
    plv_sig_connections = []

    # Separate files into different container according to algorithm
    for file in files:
        # ccorr
        if ("actual_score_data" in file and "ccorr" in file):
            ccorr_sig_connections.append(file)
            # Sort the list
            ccorr_sig_connections.sort()
        # coh
        elif ("actual_score_data" in file and "coh" in file) :
            coh_sig_connections.append(file)
            # Sort the list
            coh_sig_connections.sort()
        # plv
        elif ("actual_score_data" in file and "plv" in file) :
            plv_sig_connections.append(file)
            # Sort the list
            plv_sig_connections.sort()

    # Define list for ccorr per frequency
    total_sig_ccorr_theta_connections = []
    total_sig_ccorr_alpha_connections = []
    total_sig_ccorr_beta_connections = []
    total_sig_ccorr_gamma_connections = []

    # Define list for coh per frequency
    total_sig_coh_theta_connections = []
    total_sig_coh_alpha_connections = []
    total_sig_coh_beta_connections = []
    total_sig_coh_gamma_connections = []

    # Define list for plv per frequency
    total_sig_plv_theta_connections = []
    total_sig_plv_alpha_connections = []
    total_sig_plv_beta_connections = []
    total_sig_plv_gamma_connections = []


    # Count significant connection for ccorr algorithm and separate into 4 frequencies:
    # theta, alpha, beta, and gamma
    for file in ccorr_sig_connections:
        ccorr_file_2_read = os.path.join(path, file)
        ccorr_file = read_pickle(ccorr_file_2_read)
        
        # Theta = 0th index in the list
        sig_ccorr_theta_connections = ccorr_file[0]
        list_temp =[]
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_ccorr_theta_connections:
            pass
        else:
            for d in sig_ccorr_theta_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_ccorr_theta_connections.append(mean(list_temp))

        # Alpha = 1st index in the list
        sig_ccorr_alpha_connections = ccorr_file[1]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_ccorr_alpha_connections:
            pass
        else:
            for d in sig_ccorr_alpha_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_ccorr_alpha_connections.append(mean(list_temp))

        # Beta = 2nd index in the list
        sig_ccorr_beta_connections = ccorr_file[2]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_ccorr_beta_connections:
            pass
        else:
            for d in sig_ccorr_beta_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_ccorr_beta_connections.append(mean(list_temp))

        # Gamma = 3rd index in the list
        sig_ccorr_gamma_connections = ccorr_file[3]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_ccorr_gamma_connections:
            pass
        else:
            for d in sig_ccorr_gamma_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_ccorr_gamma_connections.append(mean(list_temp))


    # Count significant connection for coh algorithm and separate into 4 frequencies:
    # theta, alpha, beta, and gamma
    for file in coh_sig_connections:
        coh_file_2_read = os.path.join(path, file)
        coh_file = read_pickle(coh_file_2_read)
        
        # Theta = 0th index in the list
        sig_coh_theta_connections = coh_file[0]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_coh_theta_connections:
            pass
        else:
            for d in sig_coh_theta_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_coh_theta_connections.append(mean(list_temp))

        # Alpha = 1st index in the list
        sig_coh_alpha_connections = coh_file[1]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_coh_alpha_connections:
            pass
        else:
            for d in sig_coh_alpha_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_coh_alpha_connections.append(mean(list_temp))

        # Beta = 2nd index in the list
        sig_coh_beta_connections = coh_file[2]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_coh_beta_connections:
            pass
        else:
            for d in sig_coh_beta_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_coh_beta_connections.append(mean(list_temp))
        
        
        # Gamma = 3rd index in the list
        sig_coh_gamma_connections = coh_file[3]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_coh_gamma_connections:
            pass
        else:
            for d in sig_coh_gamma_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_coh_gamma_connections.append(mean(list_temp))
        

    # Count significant connection for plv algorithm and separate into 4 frequencies:
    # theta, alpha, beta, and gamma
    for file in plv_sig_connections:
        plv_file_2_read = os.path.join(path, file)
        plv_file = read_pickle(plv_file_2_read)
        
        # Theta = 0th index in the list
        sig_plv_theta_connections = plv_file[0]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_plv_theta_connections:
            pass
        else:
            for d in sig_plv_theta_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_plv_theta_connections.append(mean(list_temp))
        

        # Alpha = 1st index in the list
        sig_plv_alpha_connections = plv_file[1]
        list_temp =[]
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_plv_alpha_connections:
            pass
        else:
            for d in sig_plv_alpha_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_plv_alpha_connections.append(mean(list_temp))

        # Beta = 2nd index in the list
        sig_plv_beta_connections = plv_file[2]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_plv_beta_connections:
            pass
        else:
            for d in sig_plv_beta_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_plv_beta_connections.append(mean(list_temp))

        # Gamma = 3rd index in the list
        sig_plv_gamma_connections = plv_file[3]
        list_temp = []
        # Check if the list is empty (there is no significant connection), then skip
        if not sig_plv_gamma_connections:
            pass
        else:
            for d in sig_plv_gamma_connections:
                for key, value in d.items():
                    list_temp.append(value)
            # Average score values        
            total_sig_plv_gamma_connections.append(mean(list_temp))
        

    all_connections = results(total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
    total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
    total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections)
    
    return all_connections

# %% [markdown]
# ## Running get average actual score function

# %%
path_dir_averted_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre/"
path_dir_averted_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_post/"
path_dir_direct_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_pre/"
path_dir_direct_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_post/"
path_dir_natural_pre = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_pre/"
path_dir_natural_post = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_post/"

averted_post = average_actual_score(path_dir_averted_post)
averted_pre = average_actual_score(path_dir_averted_pre)
direct_post = average_actual_score(path_dir_direct_post)
direct_pre = average_actual_score(path_dir_direct_pre)
natural_post = average_actual_score(path_dir_natural_post)
natural_pre = average_actual_score(path_dir_natural_pre)

# print(averted_post[9])
# print(averted_pre[9])
# print("")
# print(direct_post[9])
# print(direct_pre[9])
# print("")
# print(natural_post[9])
# print(natural_pre[9])

# %% [markdown]
# ### 2.1. Find difference of average score (ccorr, coh, & plv),Pre VS Post
# NOTE : The variable name is exactly the same with section 1.1 . Make sure you run the function of get_average_score first !!. 
#
# IMPORTANT : Change no. 9 to whatever condition that you want to test. See the multiple output of get_average_score function

# %%
# Difference between PLV theta connections (post - pre) - averted eye condition

diff_averted_plv_theta = [x -y for x,y in zip(averted_post[9], averted_pre[9])]
print(F"Difference averted plv theta : {diff_averted_plv_theta}")

# Difference between PLV theta connections (post - pre) - direct eye condition

diff_direct_plv_theta = [x -y for x,y in zip(direct_post[9], direct_pre[9])]
print(F"Difference direct plv theta : {diff_direct_plv_theta}")

# Difference between PLV theta connections (post - pre) - natural eye condition

diff_natural_plv_theta = [x -y for x,y in zip(natural_post[9], natural_pre[9])]
print(F"Difference natural plv theta : {diff_natural_plv_theta}")

# %% [markdown]
# ### 2.2. Calculate statistical difference using friedman's test (non-parametric)
# NOTE : The variable name is exactly the same with section 1.2. Make sure you run the function of get_average_score first !!

# %%
# Combine the above lists and turn them into dataframe
combine_plv_theta = []
combine_plv_theta.append(diff_averted_plv_theta)
combine_plv_theta.append(diff_direct_plv_theta)
combine_plv_theta.append(diff_natural_plv_theta)

df_averted_plv_theta = pd.DataFrame(combine_plv_theta).transpose()
df_averted_plv_theta.columns = ["averted_plv_theta", "direct_plv_theta", "natural_plv_theta"]

print("PLV theta (averted vs direct vs natural)")
pg.friedman(df_averted_plv_theta)
   
# df

# %% [markdown]
# ## Testing to get average actual score

# %%
list1 =[[1,2,3,], [4,5,6,], [7,8,9,]]
list2 =[[10,20,30,], [40,50,60,], [70,80,90,]]
list3 =[[100,200,300,], [400,500,600,], [700,800,900,]]

# Combine all lists
# NOTE : Replace the value of total_list with all subjects of actual scores
# TODO: Grab all *.pkl file with keyword of actual_score then put them into one list, in this example is total_list
total_list = []
total_list.append(list1)
total_list.append(list2)
total_list.append(list3)

# Container 
total1 = []
total2 = []
total3 = []

# Grab theta / alpha / beta / gamma from all subjects (This will result in two nested lists)
for idx1, val1 in enumerate(total_list):
   for idx2, val2 in enumerate(val1):
        if (idx2 == 0):
            total1.append(val1[idx2])
        elif (idx2 == 1):
            total2.append(val1[idx2])
        elif (idx2 == 2):
            total3.append(val1[idx2])


# Put all significant connections into separate one list divided by frequency: theta, alpha, beta, and gamma
# Make 2 nested lists into one list

# Theta
total_theta_averted_pre = []
for idx1, val1 in enumerate(total1):
    for x in val1:
        total_theta_averted_pre.append(x)
print(total_theta_averted_pre)

# Alpha
total_alpha_averted_pre = []
for idx1, val1 in enumerate(total2):
    for x in val1:
        total_alpha_averted_pre.append(x)
print(total_alpha_averted_pre)

# Beta
total_beta_averted_pre = []
for idx1, val1 in enumerate(total3):
    for x in val1:
        total_beta_averted_pre.append(x)
print(total_beta_averted_pre)

# TODO: Use this code https://github.com/ihgumilar/Hyperscanning2-redesign/issues/32
# to count average significant actual score of specific connections out of all pairs (from dictionary), which have key, from each list, eg. total_beta_averted_pre


# %% [markdown]
# ## Testing to get total significant connections
# NOTE : The following codes are using the exact same variable names like above. We just added 'len' in for loop

# %%
list1 =[[1,2,3,], [4,5,6,], [7,8,9,]]
list2 =[[10,20,30,], [40,50,60,], [70,80,90,]]
list3 =[[100,200,300,], [400,500,600,], [700,800,900,]]

# Combine all lists
# NOTE : Replace the value of total_list with all subjects of actual scores
# TODO: Grab all *.pkl file with keyword of actual_score then put them into one list, in this example is total_list
total_list = []
total_list.append(list1)
total_list.append(list2)
total_list.append(list3)

# Container 
total1 = []
total2 = []
total3 = []

# Grab theta / alpha / beta / gamma from all subjects (This will result in two nested lists)
for idx1, val1 in enumerate(total_list):
    for idx2, val2 in enumerate(val1):
        if (idx2 == 0):
            total1.append(len(val1[idx2]))
        elif (idx2 == 1):
            total2.append(len(val1[idx2]))
        elif (idx2 == 2):
            total3.append(len(val1[idx2]))

print(total1)
print(total2)
print(total3)




# %% [markdown]
# ## Add leading zeros for subject number 1 - 9 
# For all algorithms : ccorr, coh, and plv

# %%
def add_leading_zero(path,N):
    # N = No. of zeros required
    files = os.listdir(path)
    new_filenames = []
    files_to_rename = []

    # Loop the filename
    for file in files:

        # Find index "S" and get 2 characters after "S"
        s_idx = file.index("S")
        subj_no = file[s_idx+1 : s_idx+3]

        # Check if the 2nd character is digit or not
        # If digit, don't include. Because we want only subject number less than 10 (satuan in indonesian language :)
        if (subj_no[-1]>="0" and subj_no[-1]<="9"):
            pass

        # If not digit, then INCLUDE ! and we want only score data 
        else : 
            if ("actual_score_data" in file):
                # This populate all subject with no. less than 11
                files_to_rename.append(file)

        
    for idx in range(0,len(files_to_rename),5):
        if (idx == 0):
            files_per_algorithm = files_to_rename[idx:idx+5]
            # print(files_per_algorithm)
        elif (idx == 5):
            files_per_algorithm = files_to_rename[idx:idx+5]
            # print(files_per_algorithm)
        elif (idx == 10):
            files_per_algorithm = files_to_rename[idx:idx+5]
            # print(files_per_algorithm)
        
        
        for idx, file_per_algorithm in enumerate(files_per_algorithm):
            if (idx==0):
                idx1 = file_per_algorithm.index(str(idx+1))
                idx2 = file_per_algorithm.index(str(idx+2))

            elif (idx==1):
                idx1 = file_per_algorithm.index(str(idx+2))
                idx2 = file_per_algorithm.index(str(idx+3))

            elif (idx==2):
                idx1 = file_per_algorithm.index(str(idx+3))
                idx2 = file_per_algorithm.index(str(idx+4))
            
            elif (idx==3):
                idx1 = file_per_algorithm.index(str(idx+4))
                idx2 = file_per_algorithm.index(str(idx+5))

            elif (idx==4):
                idx1 = file_per_algorithm.index(str(idx+5))
                idx2 = file_per_algorithm.index(str(idx+6))


            subj_no_1 = file_per_algorithm[idx1:idx1+1]
            if (idx==4):
                # Grab 10 instead of 1 for S10
                subj_no_2 = file_per_algorithm[idx2:idx2+2]
            else:
                # For participant S2 - S8 (Even subject)
                subj_no_2 = file_per_algorithm[idx2:idx2+1]

           
            # using zfill() adding leading zero
            lead_zero_1 = subj_no_1.zfill(N + len(subj_no_1))

            if (idx==4 and subj_no_2=="10"):
                # Don't add leading zero to 10 since there are already 2 digits
                lead_zero_2 = subj_no_2
            else:
                lead_zero_2 = subj_no_2.zfill(N + len(subj_no_2))


            # print result


            if (idx==0):
                file_per_algorithm = file_per_algorithm.replace(str(idx+1), lead_zero_1)
                file_per_algorithm = file_per_algorithm.replace(str(idx+2), lead_zero_2)

            elif (idx==1):

                file_per_algorithm = file_per_algorithm.replace(str(idx+2), lead_zero_1)
                file_per_algorithm = file_per_algorithm.replace(str(idx+3), lead_zero_2)

            elif (idx==2):

                file_per_algorithm = file_per_algorithm.replace(str(idx+3), lead_zero_1)
                file_per_algorithm = file_per_algorithm.replace(str(idx+4), lead_zero_2)
            
            elif (idx==3):

                file_per_algorithm = file_per_algorithm.replace(str(idx+4), lead_zero_1)
                file_per_algorithm = file_per_algorithm.replace(str(idx+5), lead_zero_2)

            elif (idx==4):
            

                file_per_algorithm = file_per_algorithm.replace(str(idx+5), lead_zero_1)
                # new_no_2 = file_per_algorithm.replace(str(idx+6), lead_zero_2)
            
            # print(file_per_algorithm)
            new_filenames.append(file_per_algorithm)

    # Replace actual filename
    for idx, old_file in enumerate(files_to_rename):
        # Get full path of old file
        old_filename_path = os.path.join(path, old_file)
        # Get full path of new file
        new_filename_path = os.path.join(path, new_filenames[idx])
        # Rename actual filename
        os.rename(old_filename_path, new_filename_path)

    print("Leading zeroes have been added to subject file number 1 - 9")
       
    


# %%
# path = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_pre_original"
# N = 1
# add_leading_zero(path,N)
