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
# ### Combine all dataframes and put in order with subject, eye gaze, and SPGQ ready for ANCOVA

# %% [markdown]
# - Combine all dataframes and put in order with subject, eye gaze, and SPGQ total score
# - SPGQ total score analyzed by Repeated Measure ANOVA
#

# %%
# Get SPGQ from first row from each dataframe and put into a list (SPGQ_all)

spgq_pre_all = []
spgq_post_all = []
copresence_pre_all = []
copresence_post_all = []
for idx in range(len(spgq_averted_post)):

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
                                "SPGQTotal_Pre" : spgq_pre_all,
                                "SPGQTotal_Post" : spgq_post_all,
                                "CoPresence_Pre" : copresence_pre_all,
                                "CoPresence_Post" : copresence_post_all})


df_all_eyes.head()
# #perform the repeated measures ANOVA (SPGQ)
# print("SPGQ Total score in post training")
# print(AnovaRM(data=df_all_eyes_post, depvar="SPGQTotal", subject="Subject", within=["EyeGaze"]).fit())

# print("CoPresence Total score in post training")
# #perform the repeated measures ANOVA (CoPresence)
# print(AnovaRM(data=df_all_eyes_post, depvar="CoPresence", subject="Subject", within=["EyeGaze"]).fit())

# %% [markdown]
# #### SPGQ Total score using pingouin package

# %%
res_spgq = pg.rm_anova(dv="SPGQTotal", within="EyeGaze", subject="Subject", data=df_all_eyes_post,
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


# %%
df_all_eyes.head()

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
