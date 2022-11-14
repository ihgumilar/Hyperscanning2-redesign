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
# ## Relevant packages

# %%
import pandas as pd

# %% [markdown]
# ## Load the file

# %%
df = pd.read_csv("/hpc/igum002/codes/Hyperscanning2-redesign/data/Demographic/demographic_exp2_redesign.csv")

# Get required columns
start_idx = df.columns.get_loc("Participant ID")
df = df.iloc[ 2:, start_idx : ]

# Convert object type to numeric (for age)
df["Age"] = pd.to_numeric(df.Age, errors="coerce")

# Reset index to zero
df = df.reset_index(drop=True)
df.head()

# %% [markdown]
# ## Plots

# %% [markdown]
# ### Gender

# %%
df["Gender"].value_counts().plot(kind="bar", xlabel="Gender", ylabel="Count", title="Number of participants by Gender");

# %% [markdown]
# ### Counts of gender

# %%
df["Gender"].value_counts()

# %% [markdown]
# ### Plot Age

# %%
df.hist(bins=10);

# %% [markdown]
# ### Statistics of age

# %%
df["Age"].describe()

# %% [markdown]
# ### Marital status
#

# %%
df["Marital Status"].value_counts().plot(kind="bar", xlabel="Marital Status", ylabel="Count", title="Number of participants by marital status");

# %% [markdown]
# ### Counts of marital status

# %%
df["Marital Status"].value_counts()

# %% [markdown]
# ### Ethnicity

# %%
df["Ethnicity"].value_counts().plot(kind="bar", xlabel="Ethnicity", ylabel="Count", title="Number of participants by Ethnicity");

# %% [markdown]
# ### Counts of ethnicity

# %%
df["Ethnicity"].value_counts()

# %% [markdown]
# ### Employment

# %%
df["Employment"].value_counts().plot(kind="bar", xlabel="Employment", ylabel="Count", title="Number of participants by Employment");

# %% [markdown]
# ### Counts of employment type

# %%
df["Employment"].value_counts()

# %% [markdown]
# ### Hand dominant

# %%
df["Hand Dominant"].value_counts().plot(kind="bar", xlabel="Hand Dominant", ylabel="Count", title="Number of participants by Hand Dominant");

# %% [markdown]
# ### Counts of hand dominant

# %%
df["Hand Dominant"].value_counts()

# %% [markdown]
# ### Eye Dominant

# %%
df["Eye Dominant"].value_counts().plot(kind="bar", xlabel="Eye Dominant", ylabel="Count", title="Number of participants by Eye Dominant");

# %% [markdown]
# ### Counts of eye dominant

# %%
df["Eye Dominant"].value_counts()

# %% [markdown]
# ### Foot/Leg Dominant

# %%
df["Foot/Leg Dominant"].value_counts().plot(kind="bar", xlabel="Foot Dominant", ylabel="Count", title="Number of participants by Foot Dominant");

# %% [markdown]
# ### Counts of foot dominant

# %%
df["Foot/Leg Dominant"].value_counts()

# %% [markdown]
# ### VR Experience

# %%
df["VR Experience"].value_counts().plot(kind="bar", xlabel="VR Experience", ylabel="Count", title="Number of participants by VR Experience");

# %% [markdown]
# ### Counts of VR experience

# %%
df["VR Experience"].value_counts()

# %% [markdown]
# ### Frequency of using VR

# %%
df["Freq. of using VR"].value_counts().plot(kind="bar", xlabel="Freq. of using VR", ylabel="Count", title="Number of participants by Freq of using VR");

# %% [markdown]
# ### Counts of frequency of using VR

# %%
df["Freq. of using VR"].value_counts()
