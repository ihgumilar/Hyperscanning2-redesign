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
#     display_name: hyperscanning2_redesign_new
#     language: python
#     name: python3
# ---

# %%
from phd_codes.EEG import stats

# %%
connections = stats.Connections()

# %% [markdown]
# # Sig connections

# %% [markdown]
# ## Path to actual scores

# %%
# path of actual scores of each eye condition

# Averted
averted_pre_actual_score_connections = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre" 
averted_post_actual_score_connections = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_post"
# Direct
direct_pre_actual_score_connections = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_pre"
direct_post_actual_score_connections = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_post"
# Natural
natural_pre_actual_score_connections = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_pre"
natural_post_actual_score_connections = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_post"

# %% [markdown]
# ## All eye conditions significant connection
# We need to calculate the number of significant connections first for each eye condition. Then we compare that number between pre and post (in the next cell)

# %%
# Averted
averted_pre_sig_connect = connections.count_sig_connections(averted_pre_actual_score_connections)
averted_post_sig_connect = connections.count_sig_connections(averted_post_actual_score_connections)

# Direct
direct_pre_sig_connect = connections.count_sig_connections(direct_pre_actual_score_connections)
direct_post_sig_connect = connections.count_sig_connections(direct_post_actual_score_connections)

# # Natural
natural_pre_sig_connect = connections.count_sig_connections(natural_pre_actual_score_connections)
natural_post_sig_connect = connections.count_sig_connections(natural_post_actual_score_connections)

# %%
averted_pre_sig_connect

# %% [markdown]
# # Diff n connections (pre vs post)
# We put the above result as an input in the following cell

# %%
diff_averted, diff_direct, diff_natural =  connections.diff_n_connections_pre_post(averted_pre_sig_connect, averted_post_sig_connect,
                                                                                   direct_pre_sig_connect, direct_post_sig_connect,
                                                                                   natural_pre_sig_connect, natural_post_sig_connect 
                                                                                    )

# %% [markdown]
# ## Extract Ccor only for theta, alpha, beta, and gamma
#
# ## REMEMBER !! Please change[3:4]
# * averted_pre_sig_connect[:1] # Theta
# * averted_pre_sig_connect[1:2] # Alpha
# * averted_pre_sig_connect[2:3] # Beta
# * averted_pre_sig_connect[3:4] # Gamma
#
# Do for the rest of the code

# %%

#### REMEMBER !! Please change[3:4]
# averted_pre_sig_connect[:1] # Theta
# averted_pre_sig_connect[1:2] # Alpha
# averted_pre_sig_connect[2:3] # Beta
# averted_pre_sig_connect[3:4] # Gamma
# Averted
averted_pre_sig_ccor_theta = [item for sublist in averted_pre_sig_connect[3:4] for item in sublist]
averted_post_sig_ccor_theta = [item for sublist in averted_post_sig_connect[3:4] for item in sublist]
# Direct
direct_pre_sig_ccor_theta = [item for sublist in direct_pre_sig_connect[3:4] for item in sublist]
direct_post_sig_ccor_theta = [item for sublist in direct_post_sig_connect[3:4] for item in sublist]
# Natural
natural_pre_sig_ccor_theta = [item for sublist in natural_pre_sig_connect[3:4] for item in sublist]
natural_post_sig_ccor_theta = [item for sublist in natural_post_sig_connect[3:4] for item in sublist]


# %%
# Format dataset before Repeated measures ANOVA

import pandas as pd
from statsmodels.stats.anova import AnovaRM

# Determine the maximum length of the lists
max_length = max(len(averted_pre_sig_ccor_theta), len(averted_post_sig_ccor_theta),
                 len(direct_pre_sig_ccor_theta), len(direct_post_sig_ccor_theta),
                 len(natural_pre_sig_ccor_theta), len(natural_post_sig_ccor_theta))

# Create a DataFrame for the data
data = {
    'pair': list(range(0, max_length)),  # Start from 1
    'averted_pre': [averted_pre_sig_ccor_theta[i] if i < len(averted_pre_sig_ccor_theta) else None for i in range(max_length)],
    'averted_post': [averted_post_sig_ccor_theta[i] if i < len(averted_post_sig_ccor_theta) else None for i in range(max_length)],
    'direct_pre': [direct_pre_sig_ccor_theta[i] if i < len(direct_pre_sig_ccor_theta) else None for i in range(max_length)],
    'direct_post': [direct_post_sig_ccor_theta[i] if i < len(direct_post_sig_ccor_theta) else None for i in range(max_length)],
    'natural_pre': [natural_pre_sig_ccor_theta[i] if i < len(natural_pre_sig_ccor_theta) else None for i in range(max_length)],
    'natural_post': [natural_post_sig_ccor_theta[i] if i < len(natural_post_sig_ccor_theta) else None for i in range(max_length)]
}


df_data = pd.DataFrame(data)

# Melt the DataFrame to long format, specifying 'pair' as the identifier variable
df_data_long = pd.melt(df_data, id_vars='pair', var_name='condition',
                       value_vars=['averted_pre', 'averted_post', 'direct_pre', 'direct_post', 'natural_pre', 'natural_post'],
                       value_name='inter_brain_connections')

# Define the desired order for the 'condition' variable
condition_order = ['averted_pre', 'averted_post', 'direct_pre', 'direct_post', 'natural_pre', 'natural_post']


# Sort the DataFrame by 'pair' for better organization
# df_data_long = df_data_long.sort_values(by='pair')

# Create categorical column 
df_data_long['condition'] = pd.Categorical(df_data_long['condition'], categories=condition_order, ordered=True)


# Sort values
df_data_long = df_data_long.sort_values(by=['pair', 'condition'])

# 1. Remove _pre/_post and rename column
df_data_long['gaze'] = df_data_long['condition'].str.replace('_pre', '').str.replace('_post', '')  

# 2. Create new condition column
df_data_long['condition'] = ['pre', 'post'] * int(len(df_data_long)/2)

# Reorder columns
df_data_long = df_data_long[['pair', 'gaze', 'condition', 'inter_brain_connections']]

print(df_data_long)

#Save to csv
# df_data_long.to_csv("gamma_ccor.csv", index=False)

# %% [markdown]
# ## Repeated measures ANOVA

# %%
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import os

# Get the file names of each frequency band
folder_path = '/hpc/igum002/codes/Hyperscanning2-redesign/data/Table_for_repeated_measure_ANOVA_hyper2_redesign'  # Replace with the path to your folder
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]


for csv_file in csv_files:
        
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert 'gaze' and 'condition' columns to numerical values
    df['gaze'] = pd.Categorical(df['gaze']).codes
    df['condition'] = pd.Categorical(df['condition']).codes

    # Perform repeated measures ANOVA
    rm_anova = AnovaRM(df, depvar='inter_brain_connections', subject='pair', within=['gaze', 'condition'])
    result = rm_anova.fit()

    # Display the ANOVA table
    print(f"****{csv_file}****")
    print("")
    print(result)

# %% [markdown]
# ## Visualization

# %%
import pandas as pd
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
import os
import matplotlib.pyplot as plt



# Function to perform repeated measures ANOVA
def run_anova(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert categorical variables to numerical
    df['gaze'] = pd.Categorical(df['gaze']).codes
    df['condition'] = pd.Categorical(df['condition']).codes

    # Perform repeated measures ANOVA
    anova_results = AnovaRM(df, 'inter_brain_connections', 'pair', within=['gaze', 'condition']).fit()

    # Display the ANOVA results
    print(f"\nANOVA Results for {file_path}:\n")
    print(anova_results)

    # Visualize the results using a seaborn factor plot
    sns.catplot(x='gaze', y='inter_brain_connections', hue='condition', data=df, kind='bar')


# Run ANOVA and visualization for each CSV file
for csv_file in csv_files:
    run_anova(csv_file)

# Show the plots
plt.show()


# %%
