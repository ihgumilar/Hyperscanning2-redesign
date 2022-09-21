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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ###  Relevant packages

# %%
### Relevant packages

import os
import pickle
import pandas as pd
import numpy as np

from collections import Counter
from LabelConverter import get_electrode_labels_connections

# %% [markdown]
# #### Just in case failed in importing LabelConverter

# %%
# If there is an error in importing LabelConter, then run the following line to change working directory
# Then run the above cell again
os.chdir("/hpc/igum002/codes/Hyperscanning2-redesign/EEG/analysis")

# %% [markdown]
# ### Load data (significant connections)

# %%
# Load the data
plv80 = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/averted_pre/Pre_coh_combined_pair_S9_and_S10_connection_data.pkl"

# It contains 4 elements (theta, alpha, beta, and gamma)
# Each element has 16 x 16 format or connections
dataplv80 = pd.read_pickle(plv80)

# %% [markdown]
# ### Get labels of significant connections and count how many occurences of each possible connections

# %% [markdown]
# #### Averted pre  : Adjust the file path and variable names

# %%

# Get significant matrix for each frequency (i.e, theta, alpha, beta, and gamma)
a_theta = dataplv80[0]
a_beta = dataplv80[1]

# Find indices where significant connections happened (ndarray type)
idx_a_theta = np.argwhere(a_theta == 1)
idx_a_beta = np.argwhere(a_beta == 1)


# Convert ndarray to tuple. This will be an input for converting indices to electrode labels, eg. FP1 - F7
a_idx_tuple_theta = tuple(map(tuple, idx_a_theta))
a_idx_tuple_beta = tuple(map(tuple, idx_a_beta))

# Populate significant electrode labels
significant_theta_connections = []
for i in range(len(a_idx_tuple_theta)):
    # Convert indices to electrode labels using my custom
    significant_theta_connections.append(
        get_electrode_labels_connections(a_idx_tuple_theta[i])
    )
    significant_theta_connections.append(
        get_electrode_labels_connections(a_idx_tuple_beta[i])
    )

# Count total number of occurences for each connections (the output is counter object)
total_connections_labels = Counter(significant_theta_connections)
# Sorted from the most common to the least one. Just put parameter inside most_common to see the 1st 3 most common values
print(
    f" Total connections (theta) using PLV (most => least common) \n {total_connections_labels.most_common()}"
)

# NOTE: In case, we want to sort from least to most common. Jus uncomment this
# list(reversed(total_connections_labels.most_common()))
