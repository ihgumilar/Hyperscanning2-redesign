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
