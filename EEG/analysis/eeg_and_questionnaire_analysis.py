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


# %% [markdown]
# ## Relevant packages

# %%
import os 
import re
import pandas as pd


# %%
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




# %% [markdown]
# ## Getting all files in a folder and filter only actual score

# %%

directory = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre_copy"
files = os.listdir(directory) 


# %%

new_plv_filename = []
files_to_rename = []

# Loop the filename
for file in files:

    # Find index "S" and get 2 characters after "S"
    s_idx = file.index("S")
    subj_no = file[s_idx+1 : s_idx+3]

    # Check if the 2nd character is digit or not
    # If digit, don't include. Because we want only subject no less than 10 (satuan in indonesian language :)
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

       

        # No. of zeros required
        N = 1

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
        
        print(file_per_algorithm)


# Replace actual filename
       
    


# %% [markdown]
# ## Replace actual name still has an issue

# %%
# Still has an issue....
## Replace actual file names
os.chdir(directory)
# dir_files2_replace = os.getcwd()
for idx, old_file_name in enumerate(files_to_rename):
    # old_file_name = os.getcwd()+"/"+val
    os.rename(old_file_name, file_per_algorithm[idx])



# %%
files_to_rename

# %% [markdown]
# ## Making leading to zero adjusted for file name 
#

# %%
############# This is to add leading zero ##########    
# Replace S5 with S05

file = 'Pre_ccorr_combined_pair_S5_and_S6_actual_score_data.pkl'    
idx1 = file.index("5", 0)
idx2 = file.index("6", 1)

subj_no_1 = file[idx1:idx1+1]
subj_no_2 = file[idx2:idx2+1]
# print(subj_no_1)
# print(subj_no_2)

# No. of zeros required
N = 1

# using zfill() adding leading zero
lead_zero_1 = subj_no_1.zfill(N + len(subj_no_1))

# print result
print("The string after adding leading zeros : " + str(lead_zero_1))

new_file = file.replace("5", lead_zero_1)

# new_file = file.replace("6", lead_zero_1)

print(new_file)





# %% [markdown]
# ## Original sample of making leading to zero

# %%
# initializing string
test_string = '5'

# printing original string
print("The original string : " + str(test_string))

# No. of zeros required
N = 1

# using zfill()
# adding leading zero
res = test_string.zfill(N + len(test_string))

# print result
print("The string after adding leading zeros : " + str(res))

# %%
# TODO Find pattern in file name, eg. S5 and S6. DO looping for this.
# TODO Get the index of that S5 and S6 or 5 / 6
# TODO Replace S5/ S6 with subject name with leading zero
#TODO Add filename with subject no. < 10 with 0 at the beginning, eg. S1 => S01 
# TODO Replace old filename with new one, then rename actual name with the new one
actual_score_files = []

for file in files:
    # actual_score_files.append(re.search(r"actual_score", file))
    if "actual_score" and "plv" in file:
        print(file)

# %% [markdown]
# ## Temporary

# %%
# Python3 code to demonstrate working of
# Convert list of dictionaries to Dictionary Value list
# Using loop
from collections import defaultdict
import numpy as np

# initializing lists
# list_temp = [{"Gfg" : 6},
# 			{"Gfg" : 8},
# 			{"Gfg" : 2},
# 			{"Gfg" : 12},
# 			{"Gfg" : 22}]

list_temp =[]
a = {"a" : 1}
aa = {"a" : 2}
b = {"b" : 1}
bb = {"b" : 3}

for i in range(4):
    list_temp.append(a)
    list_temp.append(aa)
    list_temp.append(b)
    list_temp.append(bb)

# printing original list
print("The original list : " + str(list_temp))

# using loop to get dictionaries
# defaultdict used to make default empty list
# for each key
res = defaultdict(list)
for sub in list_temp:
	for key in sub:
		res[key].append(sub[key])
	
# printing result
print("The extracted dictionary : " + str(dict(res)))

average_a = np.mean(res["a"])
average_b = np.mean(res["b"])

print(f"Average score of a : {average_a}")
print(f"Average score of b : {average_b}")

