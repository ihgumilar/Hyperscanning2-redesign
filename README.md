# Hyperscanning2-redesign

The objective of this experiment is to find whether different eye gaze directions : averted, direct, and natural, affects the inter-brain synchrony.&#x20;



Explain the task more details here later !



**Note** : A **difference** between **Hyperscanning2** and **Hyperscanning2-redesign** is the later adding questionnaire inside VR and also improved the UNITY that is used for the experiment.

In this experiment, there are three main data that will be analyzed :

1. [EEG](./#eeg)
2. [Eye Tracker](./#eye-tracker)
3. [Questionnaire](./#questionnaire)

## EEG

### Pre-processing

**Note** : The storage refers to HPC of ABI

1.  **Separate** EEG between baseline & experimetal data using this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/2e2b86ec25220141520bf86ab7d5dcbc472b8ea4) (**main** branch)

    This will extract EEG for both baseline and experimental data and save into

    &#x20;**Baseline**&#x20;

    &#x20;`/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/`

&#x20;        **Experimental**

&#x20; **``**`  ``/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/`` `**``   ** &#x20;

2\. **Combine** pre and post data (for each baseline and experimental), for all eye conditions, using this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/6848b536e7d79ddc2a99b0eff795e20f86fd1790) (**EEG-pre-processing** branch) (**Don't forget to change** end for loop from 24 (total number of files) to something (in case there is an incrincrease in the number of files). The result will be saved in the following folder

&#x20;  **Baseline**

&#x20;   ****    `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_baseline_data/raw_combined_baseline_data/`

&#x20;`` **Experimental**

&#x20;  ****   `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/raw_experimental_data/raw_combined_experimental_data`

* **`TODO`:** Change loop from 24 to whatever length of files that are available

&#x20;3\. **Clean** the above-(point no.2)-combined-EEG data for both baseline and experimental data using this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/eef8d39904c243cd7723808e575cdbe766cde360) (**Don't forget to change** end for loop from 24 (total number of files) to something (in case there is an increae in the number of files)

&#x20;    This will result in 3 files

&#x20;   `1. list_deleted_epoch_indices_averted_baseline_post.pkl`

&#x20;   `2. list_circular_correlation_scores_all_pairs_averted_post_no_filter.pkl (`**`ignore this kind of file. Not used in further analysis`**`)`

&#x20;         The **1st and 2nd files** are located here        &#x20;

&#x20;`/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/`

&#x20;   `3. preprocessed epoched files : for both baseline and experimental`

&#x20;      ```       `**`For epoched baseline data is located here`**` ```&#x20;

`/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_baseline_epoched_data/`

&#x20;      ```       `**`For experimental data is located here`**

/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed\_eeg\_data/raw\_preproc\_experiment\_epoched\_data/

&#x20;**NOTE** : This still needs to be moved manually to separated folder that is listed below accordingly so that the file that runs permutation can refer to the files that are stored in the following folder. The following path are only for **EXPERIMENTAL** data

&#x20;       **averted\_**_**pre** :_ `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/averted_pre/`

&#x20;      _       **averted\_post** :_ `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/averted_post/`

&#x20;  ``   _**direct\_pre** :_ `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/direct_pre/`

&#x20;  ``   **direct\_post**`: /hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/direct_post/`

&#x20;  **``   natural**_**\_pre** :_ `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/natural_pre/`

&#x20;  **``   natural\_post**`: /hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/pre-processed_eeg_data/raw_preproc_experiment_epoched_data/natural_post/`

* **TODO** : Update bad channels, in case the data has increased / updated

### Analysis and statistical permutation

1. **Statistical analysis** to check **if the connection is significant or not**. It saves significant connections as well as the actual score of such significant connection by using this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/4ceca7770d2bded1815168501939d43f08ad4a0b)&#x20;

&#x20;     For now, the number of permutation is **150.** The higher, the longer time to take to process the data !

&#x20;     This step will find which connection that is statistically significant for each pair (out of 256 possible connections) within four different frequencies : **theta**, **alpha**, **beta**, and **gamma.**

&#x20;      ****       This will result in 2 pkl files for each pair

&#x20;     1\. \*\_Significant connection.pkl&#x20;

&#x20;     2\. \*\_Actual score of that significant connection along with the label of connection, eg. FP1 - F7

&#x20;     The files will be stored in various folders that are available in six forms.&#x20;

&#x20;     **NOTE :** This is for **EXPERIMENTAL** data that has been permuted

&#x20;       **averted\_**_**pre** :_ /hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant\_connections/averted\_pre/

&#x20;       _        **averted\_post** :_ /hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant\_connections/averted\_post/

&#x20;  ``   _**direct\_pre** :_ `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_pre/`

&#x20;  ``   **direct\_post**`: /hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/direct_post/`

&#x20;  **``   natural**_**\_pre** :_ `/hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_pre/`

&#x20;  **``   natural\_post**`: /hpc/igum002/codes/Hyperscanning2-redesign/data/EEG/significant_connections/natural_post/`

&#x20;      **TODO:** It still needs to be moved to main branch. Once it is done, then change the commit hash that is located in the main branch

**NOTE** : 1.2. Add leading zero to subject number from 1 - 9 using a function of `add_leading_zero` which is available in this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/0375c4de54a34fd176b47165e9ee82a395d0de69) so that it would make easier to sort out the data later on.

2\. Count significant connection for each eye condition which is divided into different **frequencies** (theta, alpha, beta, and gamma) and  **algorithms** (ccorr, coh, and plv). Us a function of `total_significant_connections`  which is available in this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/0375c4de54a34fd176b47165e9ee82a395d0de69).

3\. ANCOVA for all participants once the above step is completed. ANCOVA which compares the number of significant connections between eye condition within a specific frequency.&#x20;

4\. Use this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/issues/32) to count **average significant actual score of specific connections** out of all pairs (from dictionary), which have key (out of all participants).&#x20;

&#x20;  **NOTE** : We need to populate into one container first (e.g. list) which has average score of each eye condition within a specific frequency and algorithm. This is still in progress [here](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/0375c4de54a34fd176b47165e9ee82a395d0de69)

## Eye Tracker

### Pre-processing

Code construction in progress  :tada:

### Analysis

Code construction in progress  :tada:

## Questionnaire

#### ANCOVA SPGQ & Co-Presence questionnaire

1. Calculation total score of each sub-scale of SPGQ
2. Calculation total score of SPGQ
3. Calculation total score of Co-Presence
4. ANCOVA for total score of SPGQ
5. ANCOVA for total score of Co-Presence

All the above stuff can be done via this [code](https://github.com/ihgumilar/Hyperscanning2-redesign/commit/b0c996d8f6e9dcc01445d04cccc79e27709230a4)&#x20;

