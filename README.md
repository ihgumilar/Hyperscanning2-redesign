---
description: >-
  Note : A difference between Hyperscanning2 and Hyperscanning2-redesign is the
  later is adding questionnaire inside VR and also improved the UNITY that is
  used for the experiment.
---

# Hyperscanning2-redesign

In this experiment there are three main data that will be analyzed :

1. [EEG](./#eeg)
2. [Eye Tracker](./#eye-tracker)
3. [Questionnaire](./#questionnaire)

## EEG

### Pre-processing

1. Separate EEG between baseline & experimetal data using this code f54c3a44ccd1b2e586ed60be23421e21ae0a3468
2. Combine pre and post data (for each baseline and experimental), for all eye conditions, using this code 9aa1e7aa7f3d721bdd485afedffaf1b7442f2e4c

* **ToDo:** Change loop from 16 to whatever length of files that are available)

1. Clean EEG data for both baseline and experimental data using this code ef2b503893976080800694056794df15111357f4

* **ToDo** : Update bad channels, in case the data has increased / updated

### Analysis and statistical permutation

1. Statistical analysis to check if the connection is significant or not. It saves significant connections as well as the actual score of such significant connection by using this code f9a8c6143e4a46d6f5a0a23c0e60406b0f1981a5

* **ToDo:** It still needs to be moved to main branch. Once it is done, then change the commit hash that is located in the main branch

2\. Maybe we need to create a new file that counts the total number of connections and do  ANCOVA. The code to count how many connections and actual score (out of all participants can be found here #32

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
4. Do ANCOVA for total score of SPGQ
5. Do ANCOVA for total score of Co-Presence

All the above stuff can be done via this code b0c996d8f6e9dcc01445d04cccc79e27709230a4

