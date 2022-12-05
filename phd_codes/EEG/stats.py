# ### Relevant packages
import os
from collections import namedtuple

import numpy as np
from pandas import read_pickle


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
