# ### Relevant packages
import os
from collections import namedtuple
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_pickle
from scipy.stats import pearsonr


# %%
class Connections:
    """
    Class that is related to a number of significant connections of EEG data
    """

    def count_sig_connections(self, path: str):

        """
        Objective : Count a number of significant connections for a certain eye condition, eg. averted_pre.
                    Divided into different algorithms (ccorr, coh, and plv) and frequencies (theta, alpha, beta, and gamma)

        Parameters :
                      path (str) : A path that contains *pkl file which contains actual scores of connections.
                                   Each *.pkl file will have a lenght of 4 (the order is theta, alpha, beta, and gamma)

        Outputs:
                    all_connections (namedtuple): it returns multiple values. The order is described below:

                    total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
                    total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
                    total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections,

        """

        results = namedtuple(
            "results",
            [
                "total_sig_ccorr_theta_connections",
                "total_sig_ccorr_alpha_connections",
                "total_sig_ccorr_beta_connections",
                "total_sig_ccorr_gamma_connections",
                "total_sig_coh_theta_connections",
                "total_sig_coh_alpha_connections",
                "total_sig_coh_beta_connections",
                "total_sig_coh_gamma_connections",
                "total_sig_plv_theta_connections",
                "total_sig_plv_alpha_connections",
                "total_sig_plv_beta_connections",
                "total_sig_plv_gamma_connections",
            ],
        )

        files = os.listdir(path)
        # Create new list to count the number of significant connection (eg. list_at, list_aa, list_ab, list_ag)
        ccorr_sig_connections = []
        coh_sig_connections = []
        plv_sig_connections = []

        # Separate files into different container according to algorithm
        for file in files:
            # ccorr
            if "actual_score_data" in file and "ccorr" in file:
                ccorr_sig_connections.append(file)
                # Sort the list
                ccorr_sig_connections.sort()
            # coh
            elif "actual_score_data" in file and "coh" in file:
                coh_sig_connections.append(file)
                # Sort the list
                coh_sig_connections.sort()
            # plv
            elif "actual_score_data" in file and "plv" in file:
                plv_sig_connections.append(file)
                # Sort the list
                plv_sig_connections.sort()

        # Define list for ccorr per frequency
        total_sig_ccorr_theta_connections = []
        total_sig_ccorr_alpha_connections = []
        total_sig_ccorr_beta_connections = []
        total_sig_ccorr_gamma_connections = []

        # Define list for coh per frequency
        total_sig_coh_theta_connections = []
        total_sig_coh_alpha_connections = []
        total_sig_coh_beta_connections = []
        total_sig_coh_gamma_connections = []

        # Define list for plv per frequency
        total_sig_plv_theta_connections = []
        total_sig_plv_alpha_connections = []
        total_sig_plv_beta_connections = []
        total_sig_plv_gamma_connections = []

        # Count significant connection for ccorr algorithm and separate into 4 frequencies:
        # theta, alpha, beta, and gamma
        for file in ccorr_sig_connections:
            ccorr_file_2_read = os.path.join(path, file)
            ccorr_file = read_pickle(ccorr_file_2_read)

            # Theta = 0th index in the list
            sig_ccorr_theta_connections = len(ccorr_file[0])
            total_sig_ccorr_theta_connections.append(sig_ccorr_theta_connections)

            # Alpha = 1st index in the list
            sig_ccorr_alpha_connections = len(ccorr_file[1])
            total_sig_ccorr_alpha_connections.append(sig_ccorr_alpha_connections)

            # Beta = 2nd index in the list
            sig_ccorr_beta_connections = len(ccorr_file[2])
            total_sig_ccorr_beta_connections.append(sig_ccorr_beta_connections)

            # Gamma = 3rd index in the list
            sig_ccorr_gamma_connections = len(ccorr_file[3])
            total_sig_ccorr_gamma_connections.append(sig_ccorr_gamma_connections)

        # Count significant connection for coh algorithm and separate into 4 frequencies:
        # theta, alpha, beta, and gamma
        for file in coh_sig_connections:
            coh_file_2_read = os.path.join(path, file)
            coh_file = read_pickle(coh_file_2_read)

            # Theta = 0th index in the list
            sig_coh_theta_connections = len(coh_file[0])
            total_sig_coh_theta_connections.append(sig_coh_theta_connections)

            # Alpha = 1st index in the list
            sig_coh_alpha_connections = len(coh_file[1])
            total_sig_coh_alpha_connections.append(sig_coh_alpha_connections)

            # Beta = 2nd index in the list
            sig_coh_beta_connections = len(coh_file[2])
            total_sig_coh_beta_connections.append(sig_coh_beta_connections)

            # Gamma = 3rd index in the list
            sig_coh_gamma_connections = len(coh_file[3])
            total_sig_coh_gamma_connections.append(sig_coh_gamma_connections)

        # Count significant connection for plv algorithm and separate into 4 frequencies:
        # theta, alpha, beta, and gamma
        for file in plv_sig_connections:
            plv_file_2_read = os.path.join(path, file)
            plv_file = read_pickle(plv_file_2_read)

            # Theta = 0th index in the list
            sig_plv_theta_connections = len(plv_file[0])
            total_sig_plv_theta_connections.append(sig_plv_theta_connections)

            # Alpha = 1st index in the list
            sig_plv_alpha_connections = len(plv_file[1])
            total_sig_plv_alpha_connections.append(sig_plv_alpha_connections)

            # Beta = 2nd index in the list
            sig_plv_beta_connections = len(plv_file[2])
            total_sig_plv_beta_connections.append(sig_plv_beta_connections)

            # Gamma = 3rd index in the list
            sig_plv_gamma_connections = len(plv_file[3])
            total_sig_plv_gamma_connections.append(sig_plv_gamma_connections)

        all_connections = results(
            total_sig_ccorr_theta_connections,
            total_sig_ccorr_alpha_connections,
            total_sig_ccorr_beta_connections,
            total_sig_ccorr_gamma_connections,
            total_sig_coh_theta_connections,
            total_sig_coh_alpha_connections,
            total_sig_coh_beta_connections,
            total_sig_coh_gamma_connections,
            total_sig_plv_theta_connections,
            total_sig_plv_alpha_connections,
            total_sig_plv_beta_connections,
            total_sig_plv_gamma_connections,
        )

        return all_connections

    def diff_n_connections_pre_post(
        self,
        averted_pre: tuple,
        averted_post: tuple,
        direct_pre: tuple,
        direct_post: tuple,
        natural_pre: tuple,
        natural_post: tuple,
    ):

        """
        Objective  : To find difference (absolute number) between pre and post for each eye condition, combination algorithm and frequency

        Parameters :

                     These are the results of count_sig_connections function. Run it for each eye condition
                     each result will become an input of this function.

        Outputs    :
                     - diff_averted
                     - diff_direct
                     - diff_natural

                    NOTE : Read the notes below to understand the structure of the above output of three variables

                    These are the order of list for each eye condition (diff_averted, diff_direct, diff_natural)
                    total_sig_ccorr_theta_connections, total_sig_ccorr_alpha_connections, total_sig_ccorr_beta_connections, total_sig_ccorr_gamma_connections,
                    total_sig_coh_theta_connections, total_sig_coh_alpha_connections, total_sig_coh_beta_connections, total_sig_coh_gamma_connections,
                    total_sig_plv_theta_connections, total_sig_plv_alpha_connections, total_sig_plv_beta_connections, total_sig_plv_gamma_connections

        """

        diff_averted = []
        diff_direct = []
        diff_natural = []

        for i in range(
            len(averted_pre)
        ):  # NOTE : The length is 12 means there are 12 outputs
            # that are resulted from the count_sig_connections function
            # Just pick up averted_pre variable
            diff_averted.append(
                [np.abs(x - y) for x, y in zip(averted_post[i], averted_pre[i])]
            )

            diff_direct.append(
                [np.abs(x - y) for x, y in zip(direct_post[i], direct_pre[i])]
            )

            diff_natural.append(
                [np.abs(x - y) for x, y in zip(natural_post[i], natural_pre[i])]
            )

        return diff_averted, diff_direct, diff_natural

    def corr_eeg_connection_n_question(
        self, diff_connection: List[list], diff_scale: list, title: str
    ):

        """
        Objective  : Analyze pearson correlation between number of connections of EEG
                    (substracted between post and pre) and subscale of SPGQ or SPGQ total score

        Parameters :
                    - diff_connection(List[list]) : Substracted number of connections of EEG. Each list will have six order
                                                    as follow :  Resulted from EEG.Analysis.diff_n_connections_pre_post funct

                                                diff_connect_ccorr_theta_connections, diff_connect_ccorr_alpha_connections, diff_connect_ccorr_beta_connections, diff_connect_ccorr_gamma_connections,
                                                diff_connect_coh_theta_connections, diff_connect_coh_alpha_connections, diff_connect_coh_beta_connections, diff_connect_coh_gamma_connections,
                                                diff_connect_plv_theta_connections, diff_connect_plv_alpha_connections, diff_connect_plv_beta_connections, diff_connect_plv_gamma_connections


                    - diff_scale(list) :  Substracted subscale / total score of SPGQ between pre and post

                                            - "Empathy SPGQ"
                                            - "NegativeFeelings SPGQ"
                                            - "Behavioural SPGQ"
                                            - "SPGQ Total"
                                            - "CoPresence Total"
                                            Resulted from Questionnnaire.questionnaire.diff_score_questionnaire_pre_post funct

                    - title (str)      : Title of correlation between which eye condition and subscale of questionnaire

        Output     :
                     Print Correlational score between the following connections and subscale of questionnaire (SPGQ)

                     diff_connect_ccorr_theta_connections, diff_connect_ccorr_alpha_connections, diff_connect_ccorr_beta_connections, diff_connect_ccorr_gamma_connections,
                     diff_connect_coh_theta_connections, diff_connect_coh_alpha_connections, diff_connect_coh_beta_connections, diff_connect_coh_gamma_connections,
                     diff_connect_plv_theta_connections, diff_connect_plv_alpha_connections, diff_connect_plv_beta_connections, diff_connect_plv_gamma_connections

        """

        print(title)
        for i in range(len(diff_connection)):
            print(f"{i}, {pearsonr(diff_connection[i], diff_scale)}")

    def plot_eeg_connection_n_question(
        self,
        x_axis_diff_connection: list,
        y_axis_diff_scale: list,
        title: str,
        xlabel: str,
        ylabel: str,
    ):

        """
        Objective : Plot a correlation (scatter plot) between number of connections (EEG) and
                    score of subscale of SPGQ / Co-Presence

        Parameters :
                    - x_axis_diff_connection (list) : (data for x axis) Number of connections for a certain eye conditon, algorithm, and frequency
                    - y_axis_diff_scale (list)      : (data for y axis) Score of subscale for a certain eye conditon
                                                        - "Empathy SPGQ"
                                                        - "NegativeFeelings SPGQ"
                                                        - "Behavioural SPGQ"
                                                        - "SPGQ Total"
                                                        - "CoPresence Total"
                                                        Take ONE of the lists that is resulted from EEG.Analysis.diff_n_connections_pre_post funct
                                                        as an input

                    - title (str)                   : Title for the plot
                    - xlabel (str)                  : Xlabel for the plot
                    - ylabel (str)                  : Ylabel for the plot

        Output     :
                      Plot
        """

        # adds the title
        plt.title(title)

        # plot the data
        plt.scatter(x_axis_diff_connection, y_axis_diff_scale)

        # fits the best fitting line to the data
        plt.plot(
            np.unique(x_axis_diff_connection),
            np.poly1d(np.polyfit(x_axis_diff_connection, y_axis_diff_scale, 1))(
                np.unique(x_axis_diff_connection)
            ),
            color="red",
        )

        # Labelling axes
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
