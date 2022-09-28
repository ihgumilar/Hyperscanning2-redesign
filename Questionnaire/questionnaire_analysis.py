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

from scipy.stats import ttest_rel, f_oneway
 

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
# ### Note :
# There are 2 questionnaires here that we use in the experiment : 
# * Social Presence in Gaming Questionnaire (SPGQ), which consists of 3 subscales
#     * Psychological involvement - Empathy
#     * Psychological involvement - Negative feelings
#     * Psychological involvement - Behavioral engagement
# * Co-Presence questionnaire
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
# ## T-test

# %% [markdown]
# ### SPGQ T-test averted pre vs averted post (Sig*)

# %%
spgq_averted_pre = df_averted_pre["SPGQ Total"]
spgq_averted_post = df_averted_post["SPGQ Total"]
result_averted_spgq_pre_vs_spgq_post = ttest_rel(spgq_averted_pre, spgq_averted_post)

print(f"means of SPGQ averted_pre : {np.mean(spgq_averted_pre)}")
print(f"means of SPGQ averted_post : {np.mean(spgq_averted_post)}")
print(result_averted_spgq_pre_vs_spgq_post)

# %% [markdown]
# ### SPGQ T-test direct pre vs direct post (Sig*)

# %%
spgq_direct_pre = df_direct_pre["SPGQ Total"]
spgq_direct_post = df_direct_post["SPGQ Total"]
result_direct_spgq_pre_vs_spgq_post = ttest_rel(spgq_direct_pre, spgq_direct_post)

print(f"means of SPGQ direct_pre : {np.mean(spgq_direct_pre)}")
print(f"means of SPGQ direct_post : {np.mean(spgq_direct_post)}")
print(result_direct_spgq_pre_vs_spgq_post)

# %% [markdown]
# ### SPGQ T-test natural pre vs natural post (Sig*)

# %%
spgq_natural_pre = df_natural_pre["SPGQ Total"]
spgq_natural_post = df_natural_post["SPGQ Total"]
result_direct_spgq_pre_vs_spgq_post = ttest_rel(spgq_natural_pre, spgq_natural_post)
std_natural_pre = df_natural_pre["SPGQ Total"].std()
std_natural_post = df_natural_post["SPGQ Total"].std()

print(f"means of SPGQ natural_pre : {np.mean(spgq_natural_pre)}, SD = {std_natural_pre}")
print(f"means of SPGQ natural_post : {np.mean(spgq_natural_post)}, SD = {std_natural_post}")
print(result_direct_spgq_pre_vs_spgq_post)

# %% [markdown]
# ### CoPresence T-test averted pre vs averted post

# %%
copresence_averted_pre = df_averted_pre["CoPresence Total"]
copresence_averted_post = df_averted_post["CoPresence Total"]
result_averted_copresence_pre_vs_copresence_post = ttest_rel(copresence_averted_pre, copresence_averted_post)

print(f"means of CoPresence averted_pre : {np.mean(copresence_averted_pre)}")
print(f"means of CoPresence averted_post : {np.mean(copresence_averted_post)}")
print(result_averted_copresence_pre_vs_copresence_post)

# %% [markdown]
# ### CoPresence T-test direct pre vs direct post

# %%
copresence_direct_pre = df_direct_pre["CoPresence Total"]
copresence_direct_post = df_direct_post["CoPresence Total"]
result_direct_copresence_pre_vs_copresence_post = ttest_rel(copresence_direct_pre, copresence_direct_post)

print(f"means of CoPresence direct_pre : {np.mean(copresence_direct_pre)}")
print(f"means of CoPresence direct_post : {np.mean(copresence_direct_post)}")
print(result_direct_copresence_pre_vs_copresence_post)

# %% [markdown]
# ### CoPresence T-test natural pre vs natural post (Sig*)

# %%
copresence_natural_pre = df_natural_pre["CoPresence Total"]
copresence_natural_post = df_natural_post["CoPresence Total"]
result_natural_copresence_pre_vs_copresence_post = ttest_rel(copresence_natural_pre, copresence_natural_post)

print(f"means of CoPresence natural_pre : {np.mean(copresence_natural_pre)}")
print(f"means of CoPresence natural_post : {np.mean(copresence_natural_post)}")
print(result_natural_copresence_pre_vs_copresence_post)

# %% [markdown]
# ### Empathy T-test averted pre vs averted post (Sig*)

# %%
empathy_averted_pre = df_averted_pre["Empathy SPGQ"]
empathy_averted_post = df_averted_post["Empathy SPGQ"]
result_averted_empathy_pre_vs_empathy_post = ttest_rel(empathy_averted_pre, empathy_averted_post)

print(f"means of empathy  averted_pre : {np.mean(empathy_averted_pre)}")
print(f"means of empathy  averted_post : {np.mean(empathy_averted_post)}")
print(result_averted_empathy_pre_vs_empathy_post)

# %% [markdown]
# ### Empathy T-test direct pre vs direct post (Sig*)

# %%
empathy_direct_pre = df_direct_pre["Empathy SPGQ"]
empathy_direct_post = df_direct_post["Empathy SPGQ"]
result_direct_empathy_pre_vs_empathy_post = ttest_rel(empathy_direct_pre, empathy_direct_post)

print(f"means of empathy  direct_pre : {np.mean(empathy_direct_pre)}")
print(f"means of empathy  direct_post : {np.mean(empathy_direct_post)}")
print(result_direct_empathy_pre_vs_empathy_post)

# %% [markdown]
# ### Empathy T-test natural pre vs natural post (Sig*)

# %%
empathy_natural_pre = df_natural_pre["Empathy SPGQ"]
empathy_natural_post = df_natural_post["Empathy SPGQ"]
result_natural_empathy_pre_vs_empathy_post = ttest_rel(empathy_natural_pre, empathy_natural_post)

print(f"means of empathy  natural_pre : {np.mean(empathy_natural_pre)}")
print(f"means of empathy  natural_post : {np.mean(empathy_natural_post)}")
print(result_natural_empathy_pre_vs_empathy_post)

# %% [markdown]
# ## ANOVA Eye Gaze post test (Averted, Direct, and Natural)

# %%
# Get some data of SPGQ from each post test of eye condition
averted_post_spgq = df_averted_post["SPGQ Total"]
direct_post_spgq = df_direct_post["SPGQ Total"]
natural_post_spgq = df_natural_post["SPGQ Total"]

# Conduct the one-way ANOVA
print(f"means of averted post SPGQ : {np.mean(averted_post_spgq)}")
print(f"means of direct post SPGQ : {np.mean(direct_post_spgq)}")
print(f"means of natural post SPGQ :{np.mean(natural_post_spgq)}")

f_oneway(averted_post_spgq, direct_post_spgq, natural_post_spgq)

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
