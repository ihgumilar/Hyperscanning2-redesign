import mne
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%%
def extract_eeg_data(filename, labelsequence):
    try:
        f = open(filename)
        labelsequence = int(labelsequence)

    except IOError as err_filename:
        print("The format of file name is not correct or file doesn't exist \nThe format must be 'EEG-Sx.csv' , x=subject number ")
        raise
    except ValueError as err_integer:
        print("The labelsequence input must be integer : ", err_integer)
        raise

    else:
        if  labelsequence < 1 or labelsequence > 12:
            print("The value for labelsequence parameter is out of range. It m/hpc/codes/temp_mne/mne-hypyp$ust be be between 1 and 12")
            raise IndexError
        else:

            #%% Load the data
            # filename = "EEG-S1.csv"
            fileName = filename  # The format of file name of EEG must be like this
            print("Processing file : " + fileName)
            df = pd.read_csv(fileName, delimiter=',')
            # %% Define columns for raw csv file
            df.columns = ['Index', 'FP1', 'FP2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1',
                          'O2', 'X1', 'X2', 'X3', 'X4', 'X5',
                          'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'Marker']
            # Converting all markers (pandas data frame to list)
            markers = df['Marker'].tolist()
            # %%  Find all experimental markers and print them out.

            indicesOfMarkers = []  # Empty list to contain indices of markers
            for i, c in enumerate(markers):
                if "9999999" in str(c) : #todo untuk data EEG-S1-6.csv, tambahkan " : "
                    indicesOfMarkers.append(i) # check if the number of markers = 36
            try:
                number_markers = len(indicesOfMarkers)
                if number_markers != 36:
                    raise ValueError("The {} file has incorrect number of markers : {} ! It MUST be 36".format(fileName,number_markers))
            except ValueError as err_unmatch_markers:
                print(err_unmatch_markers)
                raise


            # %% Loop the list of labels in sequence
            # todo Create a list of labels with different sequences and put them into a list (list of list)
            # Order = 1 (Averted/Direct/Natural)
            oddOrder1 = ["averted_pre_right_point", "averted_pre_left_point", "averted_left_tracking", "averted_right_tracking",
                         "averted_post_right_point", "averted_post_left_point", "direct_pre_right_point", "direct_pre_left_point",
                         "direct_left_tracking", "direct_right_tracking", "direct_post_right_point", "direct_post_left_point",
                         "natural_pre_right_point", "natural_pre_left_point", "natural_left_tracking", "natural_right_tracking",
                         "natural_post_right_point", "natural_post_left_point"]

            evenOrder1 = ["averted_pre_left_point", "averted_pre_right_point", "averted_right_tracking", "averted_left_tracking",
                          "averted_post_left_point", "averted_post_right_point", "direct_pre_left_point", "direct_pre_right_point",
                          "direct_right_tracking", "direct_left_tracking", "direct_post_left_point", "direct_post_right_point",
                          "natural_pre_left_point", "natural_pre_right_point", "natural_right_tracking", "natural_left_tracking",
                          "natural_post_left_point", "natural_post_right_point"]

            # Order = 2 (Averted/Natural/Direct)
            oddOrder2 = ["averted_pre_right_point", "averted_pre_left_point", "averted_left_tracking", "averted_right_tracking",
                         "averted_post_right_point", "averted_post_left_point", "natural_pre_right_point", "natural_pre_left_point",
                         "natural_left_tracking", "natural_right_tracking", "natural_post_right_point", "natural_post_left_point",
                         "direct_pre_right_point", "direct_pre_left_point", "direct_left_tracking", "direct_right_tracking",
                         "direct_post_right_point", "direct_post_left_point"]

            evenOrder2 = ["averted_pre_left_point", "averted_pre_right_point", "averted_right_tracking", "averted_left_tracking",
                          "averted_post_left_point", "averted_post_right_point", "natural_pre_left_point", "natural_pre_right_point",
                          "natural_right_tracking", "natural_left_tracking", "natural_post_left_point", "natural_post_right_point",
                          "direct_pre_left_point", "direct_pre_right_point", "direct_right_tracking", "direct_left_tracking",
                          "direct_post_left_point", "direct_post_right_point"]

            # Order = 3 (Direct / Natural / Averted)
            oddOrder3 = ["direct_pre_right_point", "direct_pre_left_point", "direct_left_tracking", "direct_right_tracking",
                         "direct_post_right_point", "direct_post_left_point", "natural_pre_right_point", "natural_pre_left_point",
                         "natural_left_tracking", "natural_right_tracking", "natural_post_right_point", "natural_post_left_point",
                         "averted_pre_right_point", "averted_pre_left_point", "averted_left_tracking", "averted_right_tracking",
                         "averted_post_right_point", "averted_post_left_point"]

            evenOrder3 = ["direct_pre_left_point", "direct_pre_right_point", "direct_right_tracking", "direct_left_tracking",
                          "direct_post_left_point", "direct_post_right_point", "natural_pre_left_point", "natural_pre_right_point",
                          "natural_right_tracking", "natural_left_tracking", "natural_post_left_point", "natural_post_right_point",
                          "averted_pre_left_point", "averted_pre_right_point", "averted_right_tracking", "averted_left_tracking",
                          "averted_post_left_point", "averted_post_right_point"]

            # Order = 4 (Direct/Averted/Natural)
            oddOrder4 = ["direct_pre_right_point", "direct_pre_left_point", "direct_left_tracking", "direct_right_tracking",
                         "direct_post_right_point", "direct_post_left_point", "averted_pre_right_point", "averted_pre_left_point",
                         "averted_left_tracking", "averted_right_tracking", "averted_post_right_point", "averted_post_left_point",
                         "natural_pre_right_point", "natural_pre_left_point", "natural_left_tracking", "natural_right_tracking",
                         "natural_post_right_point", "natural_post_left_point"]

            evenOrder4 = ["direct_pre_left_point", "direct_pre_right_point", "direct_right_tracking", "direct_left_tracking",
                          "direct_post_left_point", "direct_post_right_point", "averted_pre_left_point", "averted_pre_right_point",
                          "averted_right_tracking", "averted_left_tracking", "averted_post_left_point", "averted_post_right_point",
                          "natural_pre_left_point", "natural_pre_right_point", "natural_right_tracking", "natural_left_tracking",
                          "natural_post_left_point", "natural_post_right_point"]

            # Order = 5 (Natural/Direct/Averted)
            oddOrder5 = ["natural_pre_right_point", "natural_pre_left_point", "natural_left_tracking", "natural_right_tracking",
                         "natural_post_right_point", "natural_post_left_point", "direct_pre_right_point", "direct_pre_left_point",
                         "direct_left_tracking", "direct_right_tracking", "direct_post_right_point", "direct_post_left_point",
                         "averted_pre_right_point", "averted_pre_left_point", "averted_left_tracking", "averted_right_tracking",
                         "averted_post_right_point", "averted_post_left_point"]

            evenOrder5 = ["natural_pre_left_point", "natural_pre_right_point", "natural_right_tracking", "natural_left_tracking",
                          "natural_post_left_point", "natural_post_right_point", "direct_pre_left_point", "direct_pre_right_point",
                          "direct_right_tracking", "direct_left_tracking", "direct_post_left_point", "direct_post_right_point",
                          "averted_pre_left_point", "averted_pre_right_point", "averted_right_tracking", "averted_left_tracking",
                          "averted_post_left_point", "averted_post_right_point"]

            # Order = 6 (Natural/Averted/Direct)
            oddOrder6 = ["natural_pre_right_point", "natural_pre_left_point", "natural_left_tracking", "natural_right_tracking",
                         "natural_post_right_point", "natural_post_left_point", "averted_pre_right_point", "averted_pre_left_point",
                         "averted_left_tracking", "averted_right_tracking", "averted_post_right_point", "averted_post_left_point",
                         "direct_pre_right_point", "direct_pre_left_point", "direct_left_tracking", "direct_right_tracking",
                         "direct_post_right_point", "direct_post_left_point"]

            evenOrder6 = ["natural_pre_left_point", "natural_pre_right_point", "natural_right_tracking", "natural_left_tracking",
                          "natural_post_left_point", "natural_post_right_point", "averted_pre_left_point", "averted_pre_right_point",
                          "averted_right_tracking", "averted_left_tracking", "averted_post_left_point", "averted_post_right_point",
                          "direct_pre_left_point", "direct_pre_right_point", "direct_right_tracking", "direct_left_tracking",
                          "direct_post_left_point", "direct_post_right_point"]
            #%% Add all labels into a list

            listOfOrders = []
            for i in progressbar(range(6)):
                listOfOrders.append(eval('oddOrder' + str(i+1)))
                listOfOrders.append(eval('evenOrder' + str(i+1)))
            print("Data have been extracted from : " + fileName)
            chosenOrder = listOfOrders[labelsequence-1]
            # %% Chunk the data based on opening and closing markers and get only the 16 channels data (columns)
            chunkedData = []
            for i in range(0, 36, 2):
                # averted_pre_right_point = df.iloc[indicesOfMarkers[0] : indicesOfMarkers[1] +1, 1:17]
                # Change into numpy and convert it from uV (microvolts) / nV to V (volts)
                chunkedData.append(df.iloc[indicesOfMarkers[i]: indicesOfMarkers[i + 1] + 1, 1:17].to_numpy() * 1e-6)
            # %% Load each eye condition file into MNE Python
            # Create 16 channels montage 10-20 international standard
            montage = mne.channels.make_standard_montage('standard_1020')
            # % Create info
            # Pick only 16 channels that are used in Cyton+Daisy OpenBCI
            ch_names = ['FP1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'C4', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
            ch_types = ['eeg'] * 16
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=125,
                ch_types=ch_types)
            info.set_montage('standard_1020', match_case=False)
            # %% Create filenames for *.fif based on the sequence of labels above
            filenames_fif = []
            for i in chosenOrder:
                filenames_fif.append(fileName[4:fileName.index(".")] + "-" + i + "_raw.fif")
            #%% Save into *.fif files
            for i, val in enumerate(chunkedData):
                data_need_label = mne.io.RawArray(val.transpose(), info, verbose=False)
                data_need_label.save(join(filenames_fif[i]), overwrite=True)
            # todo save it into MNE-BIDS format



# extract_eeg_data("EEG-S10.csv", 1)
