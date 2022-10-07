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

# %%
# ### Correlation
# ANCOVA for significant connections

# ### Populate all significant connections for each eye condition, Different frequency has different list

# Loop for all significant connections
# e.g., Pre_ccorr_combined_pair_S1_and_S2_actual_score_data.pkl (consists of 4 lists - theta, alpha, beta, & gamma)

# Set up to directory path of significant connection, averted_pre
# Gather all files that contain _connection_data keyword and put into a list (list_a)

# Create new list to count the number of significant connection (eg. list_at, list_aa, list_ab, list_ag)
# Loop list_a
# Get the first list (e.g.theta) for each subject
# Count the lenght and put into another list (list_at)
# Get the second list (e.g.alpha) for each subject
# Count the lenght and put into another list (list_aa)
# Get the third list (e.g.beta) for each subject
# Count the lenght and put into another list (list_ab)
# Get the fourth list (e.g.gamma) for each subject
# Count the lenght and put into another list (list_ag)
# list_at, list_aa, list_ab, list_ag becomes a total length of significant connections for each eye condition, eg. averted_pre)

# Repeat the same procedure for all other eye conditions,eg. averted_post, etc

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
import pandas as pd

list_a = pd.read_pickle(
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre/Pre_plv_combined_pair_S1_and_S2_actual_score_data.pkl"
)

list_temp = list_a[0]
# list_temp.append(list_a[1])
# list_temp.append(list_a[2])
# print(list_a[0])

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



