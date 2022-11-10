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
# ### Relevant packages

# %%
# ### Relevant packages
import os
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import pingouin as pg
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import ttest_rel, f_oneway, pearsonr, spearmanr
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from collections import namedtuple
from pandas import read_pickle
from statistics import mean


# %% [markdown]
# ### List files in questionnaire folder

# %%
# ### List files in data folder
questionnaire_folder = "/hpc/igum002/codes/Hyperscanning2-redesign/data/Questionnaire/"
files = os.listdir(questionnaire_folder)

# %%
# Define list
averted_questions = []
direct_questions = []
natural_questions = []

averted_pre_questions = []
averted_post_questions = []
direct_pre_questions = []
direct_post_questions = []
natural_pre_questions = []
natural_post_questions = []

# Set index to separate pre and post questionnaire for each eye condition
begin_idx_question = 0
step_idx_question = 2

# Populate averted, direct, and natural into a separate list
for file in files:
    if "averted" in file:
        averted_questions.append(file)
    elif "direct" in file:
        direct_questions.append(file)
    else:
        natural_questions.append(file)

# Separate pre/post questionnaire into a different list (eg. averted_pre, direct_pre, natural_pre)
for idx in range(begin_idx_question, len(averted_questions), step_idx_question):

    # averted_pre
    averted_pre_questions.append(averted_questions[idx])

    # averted_post
    averted_post_questions.append(averted_questions[idx+1])

    # direct_pre
    direct_pre_questions.append(direct_questions[idx])

    # direct_post
    direct_post_questions.append(direct_questions[idx+1])

    # natural_pre
    natural_pre_questions.append(natural_questions[idx])

    # natural_post
    natural_post_questions.append(natural_questions[idx+1])



# %% [markdown]
# ### Note : Related to questionnaires
# There are 2 questionnaires here that we use in the experiment : 
# * Social Presence in Gaming Questionnaire (SPGQ), which consists of 3 subscales (Higher score, Higher Social Presence)
#     * Psychological involvement - Empathy
#     * Psychological involvement - Negative feelings
#     * Psychological involvement - Behavioral engagement
# * Co-Presence questionnaire (REMEMBER : HIGER score, indicates LESS CoPresence !!!)
# * See here for details https://docs.google.com/document/d/118ZIYY5o2bhJ6LF0fYcxDA8iinaLcn1EZ5V77zt_AeQ/edit#

# %% [markdown]
# ### Scoring for each subscale (averted_pre)
# - Combine into a single dataframe

# %%
# ### Load questrionnaire data

averted_pre_all_data_list = []
# TODO Create a loop that takes all files from the above list
for idx, file in enumerate(averted_pre_questions):

    file_to_load = questionnaire_folder + file

    df = pd.read_csv(file_to_load, sep=";",)

    # Sum subscore of empathy
    df["Empathy SPGQ"] = df["Answer"][:7].sum()

    # Sum subscore of negative feeling
    df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

    # Sum subscore of behavioral
    df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

    # Total score of SPGQ
    subscales_spgq = ["Empathy SPGQ", "NegativeFeelings SPGQ", "Behavioural SPGQ"]
    df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

    # Total score of copresence
    df["CoPresence Total"] = df["Answer"][7:13].sum()


    # Get first row and all columns
    df_clean = df.iloc[0:1, 4:]

    # Put into a list
    averted_pre_all_data_list.append(df_clean)

# Combine int a single dataframe for averted pre
df_averted_pre = pd.concat(averted_pre_all_data_list, ignore_index=True)

df_averted_pre.head()


    


# %% [markdown]
# ### Scoring for each subscale (averted_post)
# - Combine into a single dataframe

# %%
# ### Load questrionnaire data

averted_post_all_data_list = []
# TODO Create a loop that takes all files from the above list
for idx, file in enumerate(averted_post_questions):

    file_to_load = questionnaire_folder + file

    df = pd.read_csv(file_to_load, sep=";",)

    # Sum subscore of empathy
    df["Empathy SPGQ"] = df["Answer"][:7].sum()

    # Sum subscore of negative feeling
    df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

    # Sum subscore of behavioral
    df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

    # Total score of SPGQ
    subscales_spgq = ["Empathy SPGQ", "NegativeFeelings SPGQ", "Behavioural SPGQ"]
    df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

    # Total score of copresence
    df["CoPresence Total"] = df["Answer"][7:13].sum()


    # Get first row and all columns
    df_clean = df.iloc[0:1, 4:]

    # Put into a list
    averted_post_all_data_list.append(df_clean)

# Combine int a single dataframe for averted pre
df_averted_post = pd.concat(averted_post_all_data_list, ignore_index=True)

df_averted_post.head()

    

# %% [markdown]
# ### Scoring for each subscale (direct_pre)
# - Combine into a single dataframe

# %%
# ### Load questrionnaire data

direct_pre_all_data_list = []
# TODO Create a loop that takes all files from the above list
for idx, file in enumerate(direct_pre_questions):

    file_to_load = questionnaire_folder + file

    df = pd.read_csv(file_to_load, sep=";",)

    # Sum subscore of empathy
    df["Empathy SPGQ"] = df["Answer"][:7].sum()

    # Sum subscore of negative feeling
    df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

    # Sum subscore of behavioral
    df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

    # Total score of SPGQ
    subscales_spgq = ["Empathy SPGQ", "NegativeFeelings SPGQ", "Behavioural SPGQ"]
    df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

    # Total score of copresence
    df["CoPresence Total"] = df["Answer"][7:13].sum()


    # Get first row and all columns
    df_clean = df.iloc[0:1, 4:]

    # Put into a list
    direct_pre_all_data_list.append(df_clean)

# Combine int a single dataframe for averted pre
df_direct_pre = pd.concat(direct_pre_all_data_list, ignore_index=True)

df_direct_pre.head()

    

# %% [markdown]
# ### Scoring for each subscale (direct_post)
# - Combine into a single dataframe

# %%
# ### Load questrionnaire data

direct_post_all_data_list = []
# TODO Create a loop that takes all files from the above list
for idx, file in enumerate(direct_post_questions):

    file_to_load = questionnaire_folder + file

    df = pd.read_csv(file_to_load, sep=";",)

    # Sum subscore of empathy
    df["Empathy SPGQ"] = df["Answer"][:7].sum()

    # Sum subscore of negative feeling
    df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

    # Sum subscore of behavioral
    df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

    # Total score of SPGQ
    subscales_spgq = ["Empathy SPGQ", "NegativeFeelings SPGQ", "Behavioural SPGQ"]
    df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

    # Total score of copresence
    df["CoPresence Total"] = df["Answer"][7:13].sum()


    # Get first row and all columns
    df_clean = df.iloc[0:1, 4:]

    # Put into a list
    direct_post_all_data_list.append(df_clean)

# Combine int a single dataframe for averted pre
df_direct_post = pd.concat(direct_post_all_data_list, ignore_index=True)

df_direct_post.head()

    

# %% [markdown]
# ### Scoring for each subscale (natural_pre)
# - Combine into a single dataframe

# %%
# ### Load questrionnaire data

natural_pre_all_data_list = []
# TODO Create a loop that takes all files from the above list
for idx, file in enumerate(natural_pre_questions):

    file_to_load = questionnaire_folder + file

    df = pd.read_csv(file_to_load, sep=";",)

    # Sum subscore of empathy
    df["Empathy SPGQ"] = df["Answer"][:7].sum()

    # Sum subscore of negative feeling
    df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

    # Sum subscore of behavioral
    df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

    # Total score of SPGQ
    subscales_spgq = ["Empathy SPGQ", "NegativeFeelings SPGQ", "Behavioural SPGQ"]
    df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

    # Total score of copresence
    df["CoPresence Total"] = df["Answer"][7:13].sum()


    # Get first row and all columns
    df_clean = df.iloc[0:1, 4:]

    # Put into a list
    natural_pre_all_data_list.append(df_clean)

# Combine int a single dataframe for averted pre
df_natural_pre = pd.concat(natural_pre_all_data_list, ignore_index=True)

df_natural_pre.head()

    

# %% [markdown]
# ### Scoring for each subscale (natural_post)
# - Combine into a single dataframe

# %%
# ### Load questrionnaire data

natural_post_all_data_list = []
# TODO Create a loop that takes all files from the above list
for idx, file in enumerate(natural_post_questions):

    file_to_load = questionnaire_folder + file

    df = pd.read_csv(file_to_load, sep=";",)

    # Sum subscore of empathy
    df["Empathy SPGQ"] = df["Answer"][:7].sum()

    # Sum subscore of negative feeling
    df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

    # Sum subscore of behavioral
    df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

    # Total score of SPGQ
    subscales_spgq = ["Empathy SPGQ", "NegativeFeelings SPGQ", "Behavioural SPGQ"]
    df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

    # Total score of copresence
    df["CoPresence Total"] = df["Answer"][7:13].sum()


    # Get first row and all columns
    df_clean = df.iloc[0:1, 4:]

    # Put into a list
    natural_post_all_data_list.append(df_clean)

# Combine int a single dataframe for averted pre
df_natural_post = pd.concat(natural_post_all_data_list, ignore_index=True)

df_natural_post.head()

    

# %% [markdown]
# ## Combine all dataframes and put in different combinations so that it is ready for ANCOVA

# %%
# Since empathy, negative feelings and behavioral engagement are not yet extracted
# into series from dataframe, then we do it here

# Empathy pre
empathy_averted_pre =  df_averted_pre["Empathy SPGQ"]
empathy_averted_post =  df_averted_post["Empathy SPGQ"]
empathy_direct_pre =  df_direct_pre["Empathy SPGQ"]
empathy_direct_post =  df_direct_post["Empathy SPGQ"]
empathy_natural_pre =  df_natural_pre["Empathy SPGQ"]
empathy_natural_post =  df_natural_post["Empathy SPGQ"]

# Negative feelings
negative_feelings_averted_pre =  df_averted_pre["Behavioural SPGQ"]
negative_feelings_averted_post =  df_averted_post["Behavioural SPGQ"]
negative_feelings_direct_pre =  df_direct_pre["Behavioural SPGQ"]
negative_feelings_direct_post =  df_direct_post["Behavioural SPGQ"]
negative_feelings_natural_pre =  df_natural_pre["Behavioural SPGQ"]
negative_feelings_natural_post =  df_natural_post["Behavioural SPGQ"]

# Behavioural engagement
behavioural_averted_pre =  df_averted_pre["Behavioural"]
behavioural_averted_post =  df_averted_post["Behavioural"]
behavioural_direct_pre =  df_direct_pre["Behavioural"]
behavioural_direct_post =  df_direct_post["Behavioural"]
behavioural_natural_pre =  df_natural_pre["Behavioural"]
behavioural_natural_post =  df_natural_post["Behavioural"]


# Define all lists
empathy_pre_spgq_all = []
empathy_post_spgq_all = []
negative_feelings_pre_spgq_all = []
negative_feelings_post_spgq_all = []
behavioural_pre_spgq_all = []
behavioural_post_spgq_all = []
spgq_pre_all = []
spgq_post_all = []
copresence_pre_all = []
copresence_post_all = []

for idx in range(len(negative_feelings_averted_pre)):

    # Empathy Pre
    empathy_pre_spgq_all.append(empathy_averted_pre[idx])
    empathy_pre_spgq_all.append(empathy_direct_pre[idx])
    empathy_pre_spgq_all.append(empathy_natural_pre[idx])

    # Empathy Post
    empathy_post_spgq_all.append(empathy_averted_post[idx])
    empathy_post_spgq_all.append(empathy_direct_post[idx])
    empathy_post_spgq_all.append(empathy_natural_post[idx])

    # Negative feelings Pre
    negative_feelings_pre_spgq_all.append(negative_feelings_averted_pre[idx])
    negative_feelings_pre_spgq_all.append(negative_feelings_direct_pre[idx])
    negative_feelings_pre_spgq_all.append(negative_feelings_natural_pre[idx])

    # Negative feelings Post
    negative_feelings_post_spgq_all.append(negative_feelings_averted_post[idx])
    negative_feelings_post_spgq_all.append(negative_feelings_direct_post[idx])
    negative_feelings_post_spgq_all.append(negative_feelings_natural_post[idx])

    # Behavioural engagement Pre
    behavioural_pre_spgq_all.append(behavioural_averted_pre[idx])
    behavioural_pre_spgq_all.append(behavioural_direct_pre[idx])
    behavioural_pre_spgq_all.append(behavioural_natural_pre[idx])

    # Behavioural engagement Post
    behavioural_post_spgq_all.append(behavioural_averted_post[idx])
    behavioural_post_spgq_all.append(behavioural_direct_post[idx])
    behavioural_post_spgq_all.append(behavioural_natural_post[idx])
    
    # SPGQ Pre
    spgq_pre_all.append(spgq_averted_pre[idx])
    spgq_pre_all.append(spgq_direct_pre[idx])
    spgq_pre_all.append(spgq_natural_pre[idx])

    # SPGQ Post
    spgq_post_all.append(spgq_averted_post[idx])
    spgq_post_all.append(spgq_direct_post[idx])
    spgq_post_all.append(spgq_natural_post[idx])

    # CoPresence
    copresence_pre_all.append(copresence_averted_pre[idx])
    copresence_pre_all.append(copresence_direct_pre[idx])
    copresence_pre_all.append(copresence_natural_pre[idx])

    # CoPresence Post
    copresence_post_all.append(copresence_averted_post[idx])
    copresence_post_all.append(copresence_direct_post[idx])
    copresence_post_all.append(copresence_natural_post[idx])

# Create subject number
# NOTE : Adjust the number according total files of questionnaire
# Remember that index starts from 0
subject_no = list(range(1,27))
subject = np.repeat(subject_no, 3)

# Create eye gaze number
# 1 = averted post
# 2 = direct post
# 3 = natural post
# NOTE : Adjust no.26 according total files of questionnaire
eye_gaze = np.tile([1, 2, 3], 26)

df_all_eyes = pd.DataFrame({"Subject" : subject,
                            "EyeGaze" : eye_gaze,
                            "Empathy_Pre" : empathy_pre_spgq_all,
                            "Empathy_Post" : empathy_post_spgq_all,
                            "NegativeFeelings_Pre" : negative_feelings_pre_spgq_all,
                            "NegativeFeelings_Post" : negative_feelings_post_spgq_all,
                            "Behavioural_Pre" : behavioural_pre_spgq_all,
                            "Behavioural_Post" : behavioural_post_spgq_all,
                            "SPGQTotal_Pre" : spgq_pre_all,
                            "SPGQTotal_Post" : spgq_post_all,
                            "CoPresence_Pre" : copresence_pre_all,
                            "CoPresence_Post" : copresence_post_all})


df_all_eyes.head()


# %% [markdown]
# ## ANOVA

# %% [markdown]
# ### ANOVA Eye Gaze post test (Averted, Direct, & Natural)
# Parametric test.

# %% [markdown]
# ## Functions

# %% [markdown]
# ### Find total significant connections

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
# ### Run function of total significant connections

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

# print(averted_post[8])
# print(averted_pre[8])
print(len(natural_pre[8]))
print(len(natural_post[8]))
# print("")
# print(direct_post[8])
# print(direct_pre[8])
# print("")
# print(natural_post[8])
# print(natural_pre[8])

# %% [markdown]
# ### Combine all connections regardless of frequency

# %%
# Averted pre
for idx, val in enumerate(averted_pre):
    if idx == 8:
        averted_pre_theta = averted_pre[idx]
    elif idx == 9:
        averted_pre_alpha = averted_pre[idx]
    elif idx == 10:
        averted_pre_beta = averted_pre[idx]
    elif idx == 11:
        averted_pre_gamma = averted_pre[idx]

averted_pre_no_freq =  []
for idx, val in enumerate(averted_pre_theta):
    averted_pre_no_freq.append((averted_pre_theta[idx] + averted_pre_alpha[idx] + averted_pre_beta[idx] + averted_pre_gamma[idx]) / 4)
    
# Averted post
for idx, val in enumerate(averted_post):
    if idx == 8:
        averted_post_theta = averted_post[idx]
    elif idx == 9:
        averted_post_alpha = averted_post[idx]
    elif idx == 10:
        averted_post_beta = averted_post[idx]
    elif idx == 11:
        averted_post_gamma = averted_post[idx]

averted_post_no_freq =  []
for idx, val in enumerate(averted_post_theta):
    averted_post_no_freq.append((averted_post_theta[idx] + averted_post_alpha[idx] + averted_post_beta[idx] + averted_post_gamma[idx]) / 4)
    

# direct pre
for idx, val in enumerate(direct_pre):
    if idx == 8:
        direct_pre_theta = direct_pre[idx]
    elif idx == 9:
        direct_pre_alpha = direct_pre[idx]
    elif idx == 10:
        direct_pre_beta = direct_pre[idx]
    elif idx == 11:
        direct_pre_gamma = direct_pre[idx]

direct_pre_no_freq =  []
for idx, val in enumerate(direct_pre_theta):
    direct_pre_no_freq.append((direct_pre_theta[idx] + direct_pre_alpha[idx] + direct_pre_beta[idx] + direct_pre_gamma[idx]) / 4)
    
# direct post
for idx, val in enumerate(direct_post):
    if idx == 8:
        direct_post_theta = direct_post[idx]
    elif idx == 9:
        direct_post_alpha = direct_post[idx]
    elif idx == 10:
        direct_post_beta = direct_post[idx]
    elif idx == 11:
        direct_post_gamma = direct_post[idx]

direct_post_no_freq =  []
for idx, val in enumerate(direct_post_theta):
    direct_post_no_freq.append((direct_post_theta[idx] + direct_post_alpha[idx] + direct_post_beta[idx] + direct_post_gamma[idx]) / 4)
    

# natural pre
for idx, val in enumerate(natural_pre):
    if idx == 8:
        natural_pre_theta = natural_pre[idx]
    elif idx == 9:
        natural_pre_alpha = natural_pre[idx]
    elif idx == 10:
        natural_pre_beta = natural_pre[idx]
    elif idx == 11:
        natural_pre_gamma = natural_pre[idx]

natural_pre_no_freq =  []
for idx, val in enumerate(natural_pre_theta):
    natural_pre_no_freq.append((natural_pre_theta[idx] + natural_pre_alpha[idx] + natural_pre_beta[idx] + natural_pre_gamma[idx]) / 4)
    
# natural post
for idx, val in enumerate(natural_post):
    if idx == 8:
        natural_post_theta = natural_post[idx]
    elif idx == 9:
        natural_post_alpha = natural_post[idx]
    elif idx == 10:
        natural_post_beta = natural_post[idx]
    elif idx == 11:
        natural_post_gamma = natural_post[idx]

natural_post_no_freq =  []
for idx, val in enumerate(natural_post_theta):
    natural_post_no_freq.append((natural_post_theta[idx] + natural_post_alpha[idx] + natural_post_beta[idx] + natural_post_gamma[idx]) / 4)
    

# %% [markdown]
# ### Find difference of number of connections between pre and post 
# NOTE : The variable name is exactly the same with section 2.1 . Make sure you run the function of total_significant_connections first !!. 
#
# IMPORTANT : Change no. 9 to whatever condition that you want to test. See the multiple output of total_significant_connections function

# %%
# Difference between pre and post for each eye condition, combination algorithm and frequency
""" 
NOTE : Read the notes below to understand the structure of the three variables below

These are the order of list for each eye condition (diff_averted, diff_direct, diff_natural)
total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections 

"""

diff_averted = []
diff_direct = []
diff_natural = []

for i in range(len(averted_pre)): # NOTE : The length is 12 means there are 12 outputs 
                                  # that are resulted from the function significant connection
                                  # as well as get average score. Just pick up averted_pre variable
    diff_averted.append([np.abs(x -y) for x,y in zip(averted_post[i], averted_pre[i])])

    diff_direct.append([np.abs(x -y) for x,y in zip(direct_post[i], direct_pre[i])])

    diff_natural.append([np.abs(x -y) for x,y in zip(natural_post[i], natural_pre[i])])

    # Without abs
    # diff_averted.append([x -y for x,y in zip(averted_post[i], averted_pre[i])])

    # diff_direct.append([x -y for x,y in zip(direct_post[i], direct_pre[i])])

    # diff_natural.append([x -y for x,y in zip(natural_post[i], natural_pre[i])])


# %% [markdown]
# ## Correlation

# %% [markdown]
# ### Combine SPGQ Total score of subject1 & 2, etc..
#

# %%
# NOTE IMPORTANT: -2 means up to subject 26 (pair 13th) so that it will be similar to current EEG data
# later on remove -2, all data of EEG has been processed
df_averted_pre_list = list(df_averted_pre["SPGQ Total"][:-2])
df_averted_post_list = list(df_averted_post["SPGQ Total"][:-2])
df_direct_pre_list = list(df_direct_pre["SPGQ Total"][:-2])
df_direct_post_list = list(df_direct_post["SPGQ Total"][:-2])
df_natural_pre_list = list(df_natural_pre["SPGQ Total"][:-2])
df_natural_post_list = list(df_natural_post["SPGQ Total"][:-2])


df_averted_pre_combined = []
df_direct_pre_combined = []
df_natural_pre_combined = []

df_averted_post_combined = []
df_direct_post_combined = []
df_natural_post_combined = []

begin = 0
end = len(df_averted_pre_list)
step = 2
for idx in range(begin, end, step):
    # Pre conditions
    df_averted_pre_combined.append((df_averted_pre_list[idx] + df_averted_pre_list[idx+1]) / 2)
    df_direct_pre_combined.append((df_direct_pre_list[idx] + df_direct_pre_list[idx+1]) / 2)
    df_natural_pre_combined.append((df_natural_pre_list[idx] + df_natural_pre_list[idx+1]) / 2)

    # Post conditions
    df_averted_post_combined.append((df_averted_post_list[idx] + df_averted_post_list[idx+1]) / 2)
    df_direct_post_combined.append((df_direct_post_list[idx] + df_direct_post_list[idx+1]) / 2)
    df_natural_post_combined.append((df_natural_post_list[idx] + df_natural_post_list[idx+1]) / 2)


# %% [markdown]
# ### Create dataframe for the number of connections (post training only)

# %%

zipped_averted = list(zip(averted_post[-4], averted_post[-3], averted_post[-2], averted_post[-1], df_averted_post_combined ))
df_averted_post = pd.DataFrame(zipped_averted, columns=["theta_averted", "alpha_averted",
                                "beta_averted", "gamma_averted", "SPGQ_Total"])

print("Averted Post Only")
print(df_averted_post)
print("")

zipped_direct = list(zip(direct_post[-4], direct_post[-3], direct_post[-2], direct_post[-1], df_direct_post_combined ))
df_direct_post = pd.DataFrame(zipped_direct, columns=["theta_direct", "alpha_direct",
                                "beta_direct", "gamma_direct", "SPGQ_Total"])

print("Direct Post Only")
print(df_direct_post)
print("")

zipped_natural = list(zip(natural_post[-4], natural_post[-3], natural_post[-2], natural_post[-1], df_natural_post_combined ))
df_natural_post = pd.DataFrame(zipped_natural, columns=["theta_natural", "alpha_natural",
                                "beta_natural", "gamma_natural", "SPGQ_Total"])

print("Natural Post Only")
print(df_natural_post)

# %% [markdown]
# #### Correlation SPGQ and Averted Post

# %%
print("Averted")
for i in range(len(averted_post)):
    print(F"{i}, {pearsonr(averted_post[i], df_averted_post_combined)}")

# %% [markdown]
# #### Correlation SPGQ and Direct Post

# %%
print("direct")
for i in range(len(direct_post)):
    print(F"{i}, {pearsonr(direct_post[i], df_direct_post_combined)}")

# %% [markdown]
# #### Correlation SPGQ and Natural Post

# %%
print("natural")
for i in range(len(natural_post)):
    print(F"{i}, {pearsonr(natural_post[i], df_natural_post_combined)}")

# %% [markdown]
# ### Combine SPGQ Total score of subject1 & 2, etc..
# NOTE : With subtraction of post and pre

# %%
# NOTE IMPORTANT: -2 means up to subject 26 (pair 13th) so that it will be similar to current EEG data
# later on remove -2, all data of EEG has been processed
# df_averted_pre_list = list(df_averted_pre["SPGQ Total"][:-2])
# df_averted_post_list = list(df_averted_post["SPGQ Total"][:-2])
# df_direct_pre_list = list(df_direct_pre["SPGQ Total"][:-2])
# df_direct_post_list = list(df_direct_post["SPGQ Total"][:-2])
# df_natural_pre_list = list(df_natural_pre["SPGQ Total"][:-2])
# df_natural_post_list = list(df_natural_post["SPGQ Total"][:-2])

df_averted_pre_list = df_averted_pre["SPGQ Total"].tolist()
df_averted_post_list = df_averted_post["SPGQ Total"].tolist()
df_direct_pre_list = df_direct_pre["SPGQ Total"].tolist()
df_direct_post_list = df_direct_post["SPGQ Total"].tolist()
df_natural_pre_list = df_natural_pre["SPGQ Total"].tolist()
df_natural_post_list = df_natural_post["SPGQ Total"].tolist()


df_averted_pre_combined = []
df_direct_pre_combined = []
df_natural_pre_combined = []

df_averted_post_combined = []
df_direct_post_combined = []
df_natural_post_combined = []

begin = 0
end = len(df_averted_pre_list)
step = 2
for idx in range(begin, end, step):
    # Pre conditions
    df_averted_pre_combined.append((df_averted_pre_list[idx] + df_averted_pre_list[idx+1]) / 2)
    df_direct_pre_combined.append((df_direct_pre_list[idx] + df_direct_pre_list[idx+1]) / 2)
    df_natural_pre_combined.append((df_natural_pre_list[idx] + df_natural_pre_list[idx+1]) / 2)

    # Post conditions
    df_averted_post_combined.append((df_averted_post_list[idx] + df_averted_post_list[idx+1]) / 2)
    df_direct_post_combined.append((df_direct_post_list[idx] + df_direct_post_list[idx+1]) / 2)
    df_natural_post_combined.append((df_natural_post_list[idx] + df_natural_post_list[idx+1]) / 2)

# Substract post and pre score of SPGQ Total
substracted_averted = np.abs([averted_post - averted_pre for averted_post, averted_pre in zip(df_averted_post_combined, df_averted_pre_combined)])
substracted_direct = np.abs([direct_post - direct_pre for direct_post, direct_pre in zip(df_direct_post_combined, df_direct_pre_combined)])
substracted_natural = np.abs([natural_post - natural_pre for natural_post, natural_pre in zip(df_natural_post_combined, df_natural_pre_combined)])

# No absolute value
# substracted_averted = [averted_post - averted_pre for averted_post, averted_pre in zip(df_averted_post_combined, df_averted_pre_combined)]
# substracted_direct = [direct_post - direct_pre for direct_post, direct_pre in zip(df_direct_post_combined, df_direct_pre_combined)]
# substracted_natural = [natural_post - natural_pre for natural_post, natural_pre in zip(df_natural_post_combined, df_natural_pre_combined)]


# %% [markdown]
# ### Create dataframe for the number of connections and SPGQ questionnaire (taking into account pre-training connections and SPGQ score as well)
# By substracting post - pre for both number of connections as well as SPGQ questionnaire

# %%

zipped_averted_diff = list(zip(diff_averted[-4], diff_averted[-3], diff_averted[-2], diff_averted[-1], substracted_averted) )
df_averted_post_diff = pd.DataFrame(zipped_averted_diff, columns=["theta_averted", "alpha_averted",
                                "beta_averted", "gamma_averted", "SPGQ_Total"])

print("Averted Post normalized")
print(df_averted_post_diff)
print("")

zipped_direct_diff = list(zip(diff_direct[-4], diff_direct[-3], diff_direct[-2], diff_direct[-1], substracted_direct)) 
df_direct_post_diff = pd.DataFrame(zipped_direct_diff, columns=["theta_direct", "alpha_direct",
                                "beta_direct", "gamma_direct", "SPGQ_Total"])

print("direct Post normalized")
print(df_direct_post_diff)
print("")

zipped_natural_diff = list(zip(diff_natural[-4], diff_natural[-3], diff_natural[-2], diff_natural[-1], substracted_natural))
df_natural_post_diff = pd.DataFrame(zipped_natural_diff, columns=["theta_natural", "alpha_natural",
                                "beta_natural", "gamma_natural", "SPGQ_Total"])

print("natural Post normalized")
print(df_natural_post_diff)
print("")



# %% [markdown]
# ### Correlation SPGQ and Averted *

# %%
print("Averted")
for i in range(len(diff_averted)):
    print(F"{i}, {pearsonr(diff_averted[i], substracted_averted)}")

# %% [markdown]
# #### Plot PLV beta & SPGQ Total

# %%
# adds the title
plt.title('Correlation of Averted eye and SPGQ')

# plot the data
plt.scatter(diff_averted[10], substracted_averted)

# fits the best fitting line to the data
plt.plot(np.unique(diff_averted[10]),
		np.poly1d(np.polyfit(diff_averted[10], substracted_averted, 1))
		(np.unique(diff_averted[10])), color='red')

# Labelling axes
plt.xlabel('Number of connections (Beta - PLV)')
plt.ylabel('SPGQ')

# %% [markdown]
# ### Sig. Correlation SPGQ and Direct *

# %%
""" NOTE :
Significant correlation between SPGQ and Direct eye conditions (EEG) in : 
 
 Coherence
  - total_sig_coh_theta_connections
  - total_sig_coh_beta_connections
  - total_sig_coh_gamma_connections
  PLV
  - total_sig_plv_gamma_connections
"""

print("Direct")
for i in range(len(diff_direct)):
    print(F"{i}, {pearsonr(diff_direct[i], substracted_direct)}")

# %%
""" NOTE :
Significant correlation between SPGQ and Direct eye conditions (EEG) in : 
 
 Coherence
  - total_sig_coh_theta_connections
  - total_sig_coh_beta_connections
  - total_sig_coh_gamma_connections
  PLV
  - total_sig_plv_gamma_connections
"""

print("Direct")
for i in range(len(diff_direct)):
    print(F"{i}, {pearsonr(diff_direct[i], substracted_direct)}")

# %% [markdown]
# #### Plot PLV beta & SPGQ Total

# %%
# adds the title
plt.title('Correlation of Direct eye and SPGQ')

# plot the data
plt.scatter(diff_direct[10], substracted_direct)

# fits the best fitting line to the data
plt.plot(np.unique(diff_direct[10]),
		np.poly1d(np.polyfit(diff_direct[10], substracted_direct, 1))
		(np.unique(diff_direct[10])), color='red')

# Labelling axes
plt.xlabel('Number of connections (Beta - PLV)')
plt.ylabel('SPGQ')

# %% [markdown]
# #### Plot PLV gamma & SPGQ Total

# %%
# adds the title
plt.title('Correlation of Direct eye and SPGQ')

# plot the data
plt.scatter(diff_direct[11], substracted_direct)

# fits the best fitting line to the data
plt.plot(np.unique(diff_direct[11]),
		np.poly1d(np.polyfit(diff_direct[11], substracted_direct, 1))
		(np.unique(diff_direct[11])), color='red')

# Labelling axes
plt.xlabel('Number of connections (Gamma - PLV)')
plt.ylabel('SPGQ')


# %% [markdown]
# ### Correlation SPGQ and Natural

# %%
print("Natural")
for i in range(len(diff_natural)):
    print(F"{i}, {pearsonr(diff_natural[i], substracted_natural)}")

# %%
print("Natural")
for i in range(len(diff_natural)):
    print(F"{i}, {pearsonr(diff_natural[i], substracted_natural)}")

# %% [markdown]
# ### Combine Empathy SPGQ/NegativeFeelings SPGQ/Behavioural SPGQ/CoPresence Total
# You can change accordingly the keyword below \n
#
# NOTE : Becareful when we run the code below and so on. Because it has the same variable names with the above section (SPGQ Total and Eye conditions : Averted, direct, and natural)

# %%
# NOTE IMPORTANT: -2 means up to subject 26 (pair 13th) so that it will be similar to current EEG data
# later on remove -2, all data of EEG has been processed
# df_averted_pre_list = list(df_averted_pre["NegativeFeelings SPGQ"][:-2])
# df_averted_post_list = list(df_averted_post["NegativeFeelings SPGQ"][:-2])
# df_direct_pre_list = list(df_direct_pre["NegativeFeelings SPGQ"][:-2])
# df_direct_post_list = list(df_direct_post["NegativeFeelings SPGQ"][:-2])
# df_natural_pre_list = list(df_natural_pre["NegativeFeelings SPGQ"][:-2])
# df_natural_post_list = list(df_natural_post["NegativeFeelings SPGQ"][:-2])

df_averted_pre_list = list(df_averted_pre["NegativeFeelings SPGQ"])
df_averted_post_list = list(df_averted_post["NegativeFeelings SPGQ"])
df_direct_pre_list = list(df_direct_pre["NegativeFeelings SPGQ"])
df_direct_post_list = list(df_direct_post["NegativeFeelings SPGQ"])
df_natural_pre_list = list(df_natural_pre["NegativeFeelings SPGQ"])
df_natural_post_list = list(df_natural_post["NegativeFeelings SPGQ"])

df_averted_pre_combined = []
df_direct_pre_combined = []
df_natural_pre_combined = []

df_averted_post_combined = []
df_direct_post_combined = []
df_natural_post_combined = []

begin = 0
end = len(df_averted_pre_list)
step = 2
for idx in range(begin, end, step):
    # Pre conditions
    df_averted_pre_combined.append((df_averted_pre_list[idx] + df_averted_pre_list[idx+1]) / 2)
    df_direct_pre_combined.append((df_direct_pre_list[idx] + df_direct_pre_list[idx+1]) / 2)
    df_natural_pre_combined.append((df_natural_pre_list[idx] + df_natural_pre_list[idx+1]) / 2)

    # Post conditions
    df_averted_post_combined.append((df_averted_post_list[idx] + df_averted_post_list[idx+1]) / 2)
    df_direct_post_combined.append((df_direct_post_list[idx] + df_direct_post_list[idx+1]) / 2)
    df_natural_post_combined.append((df_natural_post_list[idx] + df_natural_post_list[idx+1]) / 2)

# Substract post and pre score of NegativeFeelings SPGQ
substracted_averted = [averted_post - averted_pre for averted_post, averted_pre in zip(df_averted_post_combined, df_averted_pre_combined)]
substracted_direct = [direct_post - direct_pre for direct_post, direct_pre in zip(df_direct_post_combined, df_direct_pre_combined)]
substracted_natural = [natural_post - natural_pre for natural_post, natural_pre in zip(df_natural_post_combined, df_natural_pre_combined)]


# %%
df_averted_post.columns

# %% [markdown]
# ### Correlation xxx and averted

# %%
print("Averted")
for i in range(len(diff_averted)):
    print(F"{i}, {pearsonr(diff_averted[i], substracted_averted)}")

# %% [markdown]
# ### Correlation xxx and direct

# %%
""" NOTE :
Significant correlation between SPGQ and Direct eye conditions (EEG) in : 
 
 Coherence
  - total_sig_coh_theta_connections
  - total_sig_coh_beta_connections
  - total_sig_coh_gamma_connections
  PLV
  - total_sig_plv_gamma_connections
"""

print("Direct")
for i in range(len(diff_direct)):
    print(F"{i}, {pearsonr(diff_direct[i], substracted_direct)}")

# %% [markdown]
# #### Plot PLV gamma & Behavioral SPGQ

# %%
# adds the title
plt.title('Correlation of Inter-brain connection & Behavioral SPGQ')

# plot the data
plt.scatter(diff_direct[11], substracted_direct)

# fits the best fitting line to the data
plt.plot(np.unique(diff_direct[11]),
		np.poly1d(np.polyfit(diff_direct[11], substracted_direct, 1))
		(np.unique(diff_direct[11])), color='red')

# Labelling axes
plt.xlabel('Number of connections (Gamma - PLV)')
plt.ylabel('Behavioral SPGQ')


# %% [markdown]
# ### Correlation xxx and natural

# %%
print("Natural")
for i in range(len(diff_natural)):
    print(F"{i}, {pearsonr(diff_natural[i], substracted_natural)}")

# %% [markdown]
# ## Count average significant actual score of each connection (Direct eye)

# %%

# %% [markdown]
# ## Statistical Summary

# %% [markdown]
# ### Averted Pre

# %%
print("averted pre")
df_averted_pre.describe()

# %% [markdown]
# ### Averted Post

# %%
print("averted post")
df_averted_post.describe()

# %% [markdown]
# ### Direct Pre

# %%
print("direct pre")
df_direct_pre.describe()

# %% [markdown]
# ### Direct Post

# %%
print("direct post")
df_direct_post.describe()

# %% [markdown]
# ### Natural Pre

# %%
print("natural pre")
df_natural_pre.describe()

# %% [markdown]
# ### Natural Post

# %%
print("natural post")
df_natural_post.describe()
