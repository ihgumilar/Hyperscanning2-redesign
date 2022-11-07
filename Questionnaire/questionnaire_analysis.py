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

from scipy import stats
from scipy.stats import ttest_rel, f_oneway, pearsonr
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
# ## Checking normal distribution

# %% [markdown]
# ### Averted pre & Averted post

# %%
# averted pre
print("averted pre")
print(stats.normaltest(df_averted_pre["SPGQ Total"]))
# averted post
print("averted post")
print(stats.normaltest(df_averted_post["SPGQ Total"]))
# direct pre
print("direct pre")
print(stats.normaltest(df_direct_pre["SPGQ Total"]))
# direct post - not normally distributed
print("direct post - not normally distributed")
print(stats.normaltest(df_direct_post["SPGQ Total"]))
# natural pre
print("natural pre")
print(stats.normaltest(df_natural_pre["SPGQ Total"]))
# natural post - not normally distributed
print("natural post - not normally distributed")
print(stats.normaltest(df_natural_post["SPGQ Total"]))

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
# ### Empathy T-test averted pre vs averted post

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
negative_feelings_averted_pre =  df_averted_pre["NegativeFeelings SPGQ"]
negative_feelings_averted_post =  df_averted_post["NegativeFeelings SPGQ"]
negative_feelings_direct_pre =  df_direct_pre["NegativeFeelings SPGQ"]
negative_feelings_direct_post =  df_direct_post["NegativeFeelings SPGQ"]
negative_feelings_natural_pre =  df_natural_pre["NegativeFeelings SPGQ"]
negative_feelings_natural_post =  df_natural_post["NegativeFeelings SPGQ"]

# Behavioural engagement
behavioural_averted_pre =  df_averted_pre["Behavioural SPGQ"]
behavioural_averted_post =  df_averted_post["Behavioural SPGQ"]
behavioural_direct_pre =  df_direct_pre["Behavioural SPGQ"]
behavioural_direct_post =  df_direct_post["Behavioural SPGQ"]
behavioural_natural_pre =  df_natural_pre["Behavioural SPGQ"]
behavioural_natural_post =  df_natural_post["Behavioural SPGQ"]


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
subject_no = list(range(1,25))
subject = np.repeat(subject_no, 3)

# Create eye gaze number
# 1 = averted post
# 2 = direct post
# 3 = natural post
eye_gaze = np.tile([1, 2, 3], 24)

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
# ### Friedman test Eye Gaze post test (Averted, Direct, & Natural)
# Non-parametric test. This test is alternative to repeated measures ANOVA

# %%
# Get some data of SPGQ from each post test of eye condition
averted_post_spgq = df_averted_post["SPGQ Total"]
direct_post_spgq = df_direct_post["SPGQ Total"]
natural_post_spgq = df_natural_post["SPGQ Total"]

# Conduct the one-way ANOVA
print(f"means of averted post SPGQ : {np.mean(averted_post_spgq)}")
print(f"means of direct post SPGQ : {np.mean(direct_post_spgq)}")
print(f"means of natural post SPGQ :{np.mean(natural_post_spgq)}")

stats.friedmanchisquare(averted_post_spgq, direct_post_spgq, natural_post_spgq)

# %% [markdown]
# ### SPGQ Total score using pingouin package

# %%
res_spgq = pg.rm_anova(dv="SPGQTotal_Post", within="EyeGaze", subject="Subject", data=df_all_eyes,
                   detailed=True)
print(res_spgq)                   

# %% [markdown]
# ### Repeated measure ANOVA CoPresence using pingouin (Sig*)

# %%
res_copresence = pg.rm_anova(dv="CoPresence", within="EyeGaze", subject="Subject", data=df_all_eyes_post,
                   detailed=True)
print(res_copresence)                   

# %% [markdown]
# ## ANCOVA

# %% [markdown]
# ### SPGQ Total

# %%
#perform ANCOVA
pg.ancova(data=df_all_eyes, dv='SPGQTotal_Post', covar='SPGQTotal_Pre', between='EyeGaze')


# %% [markdown]
# ### CoPResence Total

# %%
#perform ANCOVA
pg.ancova(data=df_all_eyes, dv='CoPresence_Post', covar='CoPresence_Pre', between='EyeGaze')


# %% [markdown]
# ### Empathy SPGQ

# %%
#perform ANCOVA
pg.ancova(data=df_all_eyes, dv='Empathy_Post', covar='Empathy_Pre', between='EyeGaze')

# %% [markdown]
# ### Negative feelings SPGQ

# %%
#perform ANCOVA
pg.ancova(data=df_all_eyes, dv='NegativeFeelings_Post', covar='NegativeFeelings_Pre', between='EyeGaze')

# %% [markdown]
# ### Behavioural engagement SPGQ

# %%
#perform ANCOVA
pg.ancova(data=df_all_eyes, dv='Behavioural_Post', covar='Behavioural_Pre', between='EyeGaze')

# %% [markdown]
# ## Posthoc Tests

# %% [markdown]
# ### Tukey's test CoPresence

# %%
# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df_all_eyes_post["CoPresence"],
                          groups=df_all_eyes_post['EyeGaze'],)

print(tukey)

# %% [markdown]
# ### Benjamini/Hochberg FDR correction CoPresence using pingouin (Sig*)
#

# %%
post_hoc_pingouin = pg.pairwise_tests(dv="CoPresence", within="EyeGaze", subject="Subject", data=df_all_eyes_post,
                                        padjust="fdr_bh")

print(post_hoc_pingouin)                                        

# %% [markdown]
# ### Scheffe test

# %%
sp.posthoc_scheffe(df_all_eyes_post, val_col='CoPresence', group_col='EyeGaze')


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

print(averted_post[4])
print(averted_pre[4])
print("")
print(direct_post[4])
print(direct_pre[4])
print("")
print(natural_post[4])
print(natural_pre[4])

# %% [markdown]
# ### Find difference of number of connections between pre and post (Averted)
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

for i in range(12): # NOTE : 12 means there are 12 outputs that are resulted 
                              # from the function significant connection
                              # as well as get average score
    diff_averted.append([x -y for x,y in zip(averted_post[i], averted_pre[i])])

    diff_direct.append([x -y for x,y in zip(direct_post[i], direct_pre[i])])

    diff_natural.append([x -y for x,y in zip(natural_post[i], natural_pre[i])])


# %% [markdown]
# ### Find average actual scores of each pair (ccorr, coh, and plv)

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
# ## Correlation

# %% [markdown]
# #### SPGQ & CoPresence Correlation

# %%
corr_spgq_copresence_averted_post = pearsonr(averted_post_spgq, copresence_averted_post)
print(f"Correlation SPGQ & CoPresence averted post {corr_spgq_copresence_averted_post}")
print(" ")
corr_spgq_copresence_direct_post = pearsonr(direct_post_spgq, copresence_direct_post)
print(f"Correlation SPGQ & CoPresence direct post{corr_spgq_copresence_direct_post}")
print(" ")
corr_spgq_copresence_natural_post = pearsonr(natural_post_spgq, copresence_natural_post)
print(f"Correlation SPGQ & CoPresence natural post{corr_spgq_copresence_natural_post}")


# %% [markdown]
# ### Combine SPGQ Total score of subject1 & 2, etc..

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

# Substract post and pre score of SPGQ Total
substracted_averted = [averted_post - averted_pre for averted_post, averted_pre in zip(df_averted_post_combined, df_averted_pre_combined)]
substracted_direct = [direct_post - direct_pre for direct_post, direct_pre in zip(df_direct_post_combined, df_direct_pre_combined)]
substracted_natural = [natural_post - natural_pre for natural_post, natural_pre in zip(df_natural_post_combined, df_natural_pre_combined)]


# %% [markdown]
# ### Correlation SPGQ and Averted

# %%
df_diff_averted = df_averted_post["SPGQ Total"][:-2] - df_averted_pre["SPGQ Total"][:-2]

#TODO : Still has an issue because it does not have the same length
print(diff_averted[8]) 
print(list(df_diff_averted))


pearsonr(df_diff_averted,list(diff_averted[8]))

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
