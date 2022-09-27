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

# %% [markdown]
# ### List files in questionnaire folder

# %%
# ### List files in data folder
questionnaire_folder = "/hpc/igum002/codes/Hyperscanning2-redesign/data/Questionnaire"
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
# ### Scoring for each subscale

# %%
# ### Load questrionnaire data
df = pd.read_csv(
    "/hpc/igum002/codes/Hyperscanning2-redesign/data/Questionnaire/S01_avertedEyesquestionnaire_2022-06-27_T_15-01-07.csv",
    sep=";",
)

# Sum subscore of empathy
df["Empathy_spgq"] = df["Answer"][:7].sum()

# Sum subscore of negative feeling
df["NegativeFeelings_spgq"] = df["Answer"][7:13].sum()

# Sum subscore of behavioral
df["Behavioural_spgq"] = df["Answer"][13:21].sum()

# Sum total of copresence
df["CoPresence_Total"] = df["Answer"][7:13].sum()

df.head()


# %% [markdown]
# Filter data that contains empathy
# Add more keyword for filtering
# df_empathy = df[df["QuestionID"].str.contains("empathy")]
# df["Empathy"] = df.iloc[0:7]
