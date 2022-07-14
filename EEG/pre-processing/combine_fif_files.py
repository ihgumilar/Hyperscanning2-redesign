import mne
from tqdm import tqdm

# # %%
# for i in range(0,32,2):
# # Pre-averted
#     averted_pre_right_odd_subject = mne.io.read_raw_fif("S" + str(i+1) + "-averted_pre_right_point_raw.fif", verbose=False)
#     averted_pre_left_odd_subject = mne.io.read_raw_fif("S" + str(i+1) + "-averted_pre_left_point_raw.fif", verbose=False)
#     averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
#     combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
#     combined_pre_averted_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i+1) + "-averted_pre_right_left_point_combined_raw.fif"
#     combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

#%% Odd subjects !!

for i in tqdm(range(0,32,2), desc="Just relax and drink your coffee.."):
# Pre-averted
    averted_pre_right_odd_subject = mne.io.read_raw_fif("S" + str(i+1) + "-averted_pre_right_point_raw.fif", verbose=False)
    averted_pre_left_odd_subject = mne.io.read_raw_fif("S" + str(i+1) + "-averted_pre_left_point_raw.fif", verbose=False)
    averted_pre_files_to_combine = [averted_pre_right_odd_subject, averted_pre_left_odd_subject]
    combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
    combined_pre_averted_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i+1) + "-averted_pre_right_left_point_combined_raw.fif"
    combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

# Post-averted
    averted_post_right_odd_subject = mne.io.read_raw_fif("S" + str(i+1) + "-averted_post_right_point_raw.fif", verbose=False)
    averted_post_left_odd_subject = mne.io.read_raw_fif("S" + str(i+1) + "-averted_post_left_point_raw.fif", verbose=False)
    averted_post_files_to_combine = [averted_post_right_odd_subject, averted_post_left_odd_subject]
    combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
    combined_post_averted_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i+1) + "-averted_post_right_left_point_combined_raw.fif"
    combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

#  Pre-directed
    direct_pre_right_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_pre_right_point_raw.fif",verbose=False)
    direct_pre_left_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_pre_left_point_raw.fif",verbose=False)
    direct_pre_files_to_combine = [direct_pre_right_odd_subject, direct_pre_left_odd_subject]
    combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
    combined_pre_direct_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-direct_pre_right_left_point_combined_raw.fif"
    combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)

#  Post-directed
    direct_post_right_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_post_right_point_raw.fif", verbose=False)
    direct_post_left_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_post_left_point_raw.fif", verbose=False)
    direct_post_files_to_combine = [direct_post_right_odd_subject, direct_post_left_odd_subject]
    combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
    combined_post_direct_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-direct_post_right_left_point_combined_raw.fif"
    combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

#  Pre-natural
    natural_pre_right_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_pre_right_point_raw.fif", verbose=False)
    natural_pre_left_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_pre_left_point_raw.fif", verbose=False)
    natural_pre_files_to_combine = [natural_pre_right_odd_subject, natural_pre_left_odd_subject]
    combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
    combined_pre_natural_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-natural_pre_right_left_point_combined_raw.fif"
    combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)

#  Post-natural
    natural_post_right_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_post_right_point_raw.fif", verbose=False)
    natural_post_left_odd_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_post_left_point_raw.fif", verbose=False)
    natural_post_files_to_combine = [natural_post_right_odd_subject, natural_post_left_odd_subject]
    combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
    combined_post_natural_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-natural_post_right_left_point_combined_raw.fif"
    combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)

print("You files have combined, sir !. Just continue your coffee :)")

#%% Even subjects

for i in tqdm(range(1,32,2), desc="Just relax and drink your tea.."):
# Pre-averted
    averted_pre_right_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-averted_pre_right_point_raw.fif", verbose=False)
    averted_pre_left_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-averted_pre_left_point_raw.fif", verbose=False)
    averted_pre_files_to_combine = [averted_pre_left_even_subject, averted_pre_right_even_subject]
    combined_pre_averted_files = mne.concatenate_raws(averted_pre_files_to_combine)
    combined_pre_averted_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-averted_pre_left_right_point_combined_raw.fif"
    combined_pre_averted_files.save(combined_pre_averted_files_label, overwrite=True)

# Post-averted
    averted_post_right_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-averted_post_right_point_raw.fif", verbose=False)
    averted_post_left_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-averted_post_left_point_raw.fif", verbose=False)
    averted_post_files_to_combine = [averted_post_left_even_subject, averted_post_right_even_subject]
    combined_post_averted_files = mne.concatenate_raws(averted_post_files_to_combine)
    combined_post_averted_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-averted_post_left_right_point_combined_raw.fif"
    combined_post_averted_files.save(combined_post_averted_files_label, overwrite=True)

# Pre-direct
    direct_pre_right_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_pre_right_point_raw.fif", verbose=False)
    direct_pre_left_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_pre_left_point_raw.fif", verbose=False)
    direct_pre_files_to_combine = [direct_pre_left_even_subject, direct_pre_right_even_subject]
    combined_pre_direct_files = mne.concatenate_raws(direct_pre_files_to_combine)
    combined_pre_direct_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-direct_pre_left_right_point_combined_raw.fif"
    combined_pre_direct_files.save(combined_pre_direct_files_label, overwrite=True)
# Post-direct
    direct_post_right_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_post_right_point_raw.fif", verbose=False)
    direct_post_left_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-direct_post_left_point_raw.fif", verbose=False)
    direct_post_files_to_combine = [direct_post_left_even_subject, direct_post_right_even_subject]
    combined_post_direct_files = mne.concatenate_raws(direct_post_files_to_combine)
    combined_post_direct_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-direct_post_left_right_point_combined_raw.fif"
    combined_post_direct_files.save(combined_post_direct_files_label, overwrite=True)

# Pre-natural
    natural_pre_right_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_pre_right_point_raw.fif", verbose=False)
    natural_pre_left_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_pre_left_point_raw.fif", verbose=False)
    natural_pre_files_to_combine = [natural_pre_left_even_subject, natural_pre_right_even_subject]
    combined_pre_natural_files = mne.concatenate_raws(natural_pre_files_to_combine)
    combined_pre_natural_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-natural_pre_left_right_point_combined_raw.fif"
    combined_pre_natural_files.save(combined_pre_natural_files_label, overwrite=True)
# Post-natural
    natural_post_right_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_post_right_point_raw.fif", verbose=False)
    natural_post_left_even_subject = mne.io.read_raw_fif("S" + str(i + 1) + "-natural_post_left_point_raw.fif", verbose=False)
    natural_post_files_to_combine = [natural_post_left_even_subject, natural_post_right_even_subject]
    combined_post_natural_files = mne.concatenate_raws(natural_post_files_to_combine)
    combined_post_natural_files_label = "/hpc/igum002/codes/frontiers_hyperscanning2/eeg_data_combined_fif/" + "S" + str(i + 1) + "-natural_post_left_right_point_combined_raw.fif"
    combined_post_natural_files.save(combined_post_natural_files_label, overwrite=True)

print("You files have combined, bro !. Have more tea, please :)")
