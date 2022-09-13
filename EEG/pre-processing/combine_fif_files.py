# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
# ## Relevant packages

# %%
import mne
from tqdm import tqdm

# %% [markdown]
# ## Experimental data

# %% [markdown]
# ### Define paths and variables

# %%
# Go to a directory that stores raw fif file (not combined files)
raw_experimental_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/"
# Directory where to save combined fif files
combined_experimental_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data/"
# Change working directory
os.chdir(raw_experimental_dir)

# %% [markdown]
# #### Odd subjects (1-9)

# %% Odd subjects !!
begin = 0
end = 9
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
       
    # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_experimental_dir + "S0" + str(i+1) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

    # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_experimental_dir + "S0" + str(i+1) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

    #  Pre-directed
        direct_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-direct_pre_right_point_raw.fif",verbose=False)
        direct_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-direct_pre_left_point_raw.fif",verbose=False)
        direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
        combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
        combined_pre_direct_files_label = combined_experimental_dir + "S0" + str(i+1) + "-direct_pre_right_left_point_combined_raw.fif"
        combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)

    #  Post-directed
        direct_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-direct_post_right_point_raw.fif", verbose=False)
        direct_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-direct_post_left_point_raw.fif", verbose=False)
        direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
        combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
        combined_post_direct_files_label = combined_experimental_dir + "S0" + str(i+1) + "-direct_post_right_left_point_combined_raw.fif"
        combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

    #  Pre-natural
        natural_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-natural_pre_right_point_raw.fif", verbose=False)
        natural_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-natural_pre_left_point_raw.fif", verbose=False)
        natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
        combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
        combined_pre_natural_files_label = combined_experimental_dir + "S0" + str(i+1) + "-natural_pre_right_left_point_combined_raw.fif"
        combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)

    #  Post-natural
        natural_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-natural_post_right_point_raw.fif", verbose=False)
        natural_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-natural_post_left_point_raw.fif", verbose=False)
        natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
        combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
        combined_post_natural_files_label = combined_experimental_dir + "S0" + str(i+1) + "-natural_post_right_left_point_combined_raw.fif"
        combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)



print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# #### Even subjects (2-8)

# %%
begin = 0
end = 8
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
       
    # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_experimental_dir + "S0" + str(i+2) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

    # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_experimental_dir + "S0" + str(i+2) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

    #  Pre-directed
        direct_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-direct_pre_right_point_raw.fif",verbose=False)
        direct_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-direct_pre_left_point_raw.fif",verbose=False)
        direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
        combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
        combined_pre_direct_files_label = combined_experimental_dir + "S0" + str(i+2) + "-direct_pre_right_left_point_combined_raw.fif"
        combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)

    #  Post-directed
        direct_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-direct_post_right_point_raw.fif", verbose=False)
        direct_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-direct_post_left_point_raw.fif", verbose=False)
        direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
        combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
        combined_post_direct_files_label = combined_experimental_dir + "S0" + str(i+2) + "-direct_post_right_left_point_combined_raw.fif"
        combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

    #  Pre-natural
        natural_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-natural_pre_right_point_raw.fif", verbose=False)
        natural_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-natural_pre_left_point_raw.fif", verbose=False)
        natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
        combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
        combined_pre_natural_files_label = combined_experimental_dir + "S0" + str(i+2) + "-natural_pre_right_left_point_combined_raw.fif"
        combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)

    #  Post-natural
        natural_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-natural_post_right_point_raw.fif", verbose=False)
        natural_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-natural_post_left_point_raw.fif", verbose=False)
        natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
        combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
        combined_post_natural_files_label = combined_experimental_dir + "S0" + str(i+2) + "-natural_post_right_left_point_combined_raw.fif"
        combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)



print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# #### Even subjects (10 and onwards, eg. 10, 12, 14, etc..)

# %%
begin = 10
end = 16
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
    
    # Grab only file No. 10 and combine
    if (i==10):
        # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_experimental_dir + "S" + str(i) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

        # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_experimental_dir + "S" + str(i) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

        # Pre-directed
        direct_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-direct_pre_right_point_raw.fif",verbose=False)
        direct_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-direct_pre_left_point_raw.fif",verbose=False)
        direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
        combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
        combined_pre_direct_files_label = combined_experimental_dir + "S" + str(i) + "-direct_pre_right_left_point_combined_raw.fif"
        combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)

        # Post-directed
        direct_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-direct_post_right_point_raw.fif", verbose=False)
        direct_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-direct_post_left_point_raw.fif", verbose=False)
        direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
        combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
        combined_post_direct_files_label = combined_experimental_dir + "S" + str(i) + "-direct_post_right_left_point_combined_raw.fif"
        combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

        # Pre-natural
        natural_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-natural_pre_right_point_raw.fif", verbose=False)
        natural_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-natural_pre_left_point_raw.fif", verbose=False)
        natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
        combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
        combined_pre_natural_files_label = combined_experimental_dir + "S" + str(i) + "-natural_pre_right_left_point_combined_raw.fif"
        combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)

        # Post-natural
        natural_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-natural_post_right_point_raw.fif", verbose=False)
        natural_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-natural_post_left_point_raw.fif", verbose=False)
        natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
        combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
        combined_post_natural_files_label = combined_experimental_dir + "S" + str(i) + "-natural_post_right_left_point_combined_raw.fif"
        combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)

    # Combine file no.12, 14, etc..
    if (i+2 > 10):
        # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_experimental_dir + "S" + str(i+2) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

        # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_experimental_dir + "S" + str(i+2) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

        #  Pre-directed
        direct_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-direct_pre_right_point_raw.fif",verbose=False)
        direct_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-direct_pre_left_point_raw.fif",verbose=False)
        direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
        combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
        combined_pre_direct_files_label = combined_experimental_dir + "S" + str(i+2) + "-direct_pre_right_left_point_combined_raw.fif"
        combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)

        #  Post-directed
        direct_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-direct_post_right_point_raw.fif", verbose=False)
        direct_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-direct_post_left_point_raw.fif", verbose=False)
        direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
        combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
        combined_post_direct_files_label = combined_experimental_dir + "S" + str(i+2) + "-direct_post_right_left_point_combined_raw.fif"
        combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

        #  Pre-natural
        natural_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-natural_pre_right_point_raw.fif", verbose=False)
        natural_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-natural_pre_left_point_raw.fif", verbose=False)
        natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
        combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
        combined_pre_natural_files_label = combined_experimental_dir + "S" + str(i+2) + "-natural_pre_right_left_point_combined_raw.fif"
        combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)

        #  Post-natural
        natural_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-natural_post_right_point_raw.fif", verbose=False)
        natural_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-natural_post_left_point_raw.fif", verbose=False)
        natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
        combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
        combined_post_natural_files_label = combined_experimental_dir + "S" + str(i+2) + "-natural_post_right_left_point_combined_raw.fif"
        combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)



print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# #### Odd subjects (11 and onwards, eg. 11, 13, 15, etc...)

# %%
begin = 10
end = 16
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
       
    # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_experimental_dir + "S" + str(i+1) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

    # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_experimental_dir + "S" + str(i+1) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

    #  Pre-directed
        direct_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-direct_pre_right_point_raw.fif",verbose=False)
        direct_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-direct_pre_left_point_raw.fif",verbose=False)
        direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
        combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
        combined_pre_direct_files_label = combined_experimental_dir + "S" + str(i+1) + "-direct_pre_right_left_point_combined_raw.fif"
        combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)

    #  Post-directed
        direct_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-direct_post_right_point_raw.fif", verbose=False)
        direct_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-direct_post_left_point_raw.fif", verbose=False)
        direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
        combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
        combined_post_direct_files_label = combined_experimental_dir + "S" + str(i+1) + "-direct_post_right_left_point_combined_raw.fif"
        combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

    #  Pre-natural
        natural_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-natural_pre_right_point_raw.fif", verbose=False)
        natural_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-natural_pre_left_point_raw.fif", verbose=False)
        natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
        combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
        combined_pre_natural_files_label = combined_experimental_dir + "S" + str(i+1) + "-natural_pre_right_left_point_combined_raw.fif"
        combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)

    #  Post-natural
        natural_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-natural_post_right_point_raw.fif", verbose=False)
        natural_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-natural_post_left_point_raw.fif", verbose=False)
        natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
        combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
        combined_post_natural_files_label = combined_experimental_dir + "S" + str(i+1) + "-natural_post_right_left_point_combined_raw.fif"
        combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)



print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# ## Baseline data

# %% [markdown]
# #### Define path and variables

# %%
# Go to a directory that stores raw fif file (not combined files)
raw_baseline_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/"
# Directory where to save combined fif files
combined_baseline_dir = "/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/"
# Change working directory
os.chdir(raw_baseline_dir)

# %% [markdown]
# #### Odd subjects (1-9)

# %%
begin = 0
end = 9
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
       
    # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_baseline_dir + "S0" + str(i+1) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

    # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+1) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_baseline_dir + "S0" + str(i+1) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# #### Even subjects (2-8)

# %%
begin = 0
end = 8
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
       
    # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_baseline_dir + "S0" + str(i+2) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

    # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S0" + str(i+2) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_baseline_dir + "S0" + str(i+2) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# #### Even subjects (10 and onwards, eg. 10, 12, 14, etc..)

# %%
begin = 10
end = 16
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
    
    # Grab only file No. 10 and combine
    if (i==10):
        # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_baseline_dir + "S" + str(i) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

        # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_baseline_dir + "S" + str(i) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

    # Combine file no.12, 14, etc..
    if (i+2 > 10):
        # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_baseline_dir + "S" + str(i+2) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

        # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+2) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_baseline_dir + "S" + str(i+2) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

print("You files have combined, sir !. Just continue your coffee :)")

# %% [markdown]
# #### Odd subjects (11 and onwards, eg. 11, 13, 15, etc...)

# %%
begin = 10
end = 16
step = 2

for i in tqdm(range(begin,end,step), desc="Just relax and drink your coffee.."):
       
    # Pre-averted
        averted_pre_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_pre_right_point_raw.fif", verbose=False)
        averted_pre_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_pre_left_point_raw.fif", verbose=False)
        averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
        combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
        combined_pre_averted_files_label = combined_baseline_dir + "S" + str(i+1) + "-averted_pre_right_left_point_combined_raw.fif"
        combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

    # Post-averted
        averted_post_right_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_post_right_point_raw.fif", verbose=False)
        averted_post_left_odd_subject = mne.io.read_raw_fif("EEG-S" + str(i+1) + "-averted_post_left_point_raw.fif", verbose=False)
        averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
        combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
        combined_post_averted_files_label = combined_baseline_dir + "S" + str(i+1) + "-averted_post_right_left_point_combined_raw.fif"
        combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)
 
print("You files have combined, sir !. Just continue your coffee :)")
