# ### Relevant packages
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas import DataFrame


class Questionnaire:
    """
    This class contains functions that are related to questionnaire scoring.
    """

    def scoring_questionnaire(self, path2questions: str):

        """
            Scoring questionnnaire for each eye condition. Both subscales and\
            total score of Social Prensece Game Questionnaire (SPGQ)

            :param path2questions: Path to where raw questionnaire data (*.csv) is stored.
            :type path2questions: str
            :return all_questionnaires: all scored questionnaires for each eye condition
            :rtype: namedtuple
            

            .. note::
                returns:
                    * all_questionnaires (namedtuple). Here is the order :
                        * "averted_pre"
                        * "averted_post"
                        * "direct_pre"
                        * "direct_post"
                        * "natural_pre"
                        * "natural_post"
                        
                questionnaires:
                    * There are 2 questionnaires here that we use in the experiment :
                    * Social Presence in Gaming Questionnaire (SPGQ), which consists of 3 subscales\
                      (Higher score, Higher Social Presence)
                        * Psychological involvement - Empathy
                        * Psychological involvement - Negative feelings
                        * Psychological involvement - Behavioral engagement

                    * Co-Presence questionnaire (REMEMBER : HIGHER score, indicates LESS CoPresence !!!)
                    * See `here for details <https://docs.google.com/document/d/118ZIYY5o2bhJ6LF0fYcxDA8iinaLcn1EZ5V77zt_AeQ/edit#>`_.

        """

        results = namedtuple(
            "questionnnaires",
            [
                "averted_pre",
                "averted_post",
                "direct_pre",
                "direct_post",
                "natural_pre",
                "natural_post",
            ],
        )

        # Define list

        files = os.listdir(path2questions)
        averted_questions = []
        direct_questions = []
        natural_questions = []

        averted_pre_questions = []
        averted_post_questions = []
        direct_pre_questions = []
        direct_post_questions = []
        natural_pre_questions = []
        natural_post_questions = []

        # Set index to separate pre and post questionnaire for each eye condition
        begin_idx_question = 0
        step_idx_question = 2

        # Populate averted, direct, and natural into a separate list
        for file in files:
            if "averted" in file:
                averted_questions.append(file)
            elif "direct" in file:
                direct_questions.append(file)
            else:
                natural_questions.append(file)

        # Separate pre/post questionnaire into a different list (eg. averted_pre, direct_pre, natural_pre)
        for idx in range(begin_idx_question, len(averted_questions), step_idx_question):

            # averted_pre
            averted_pre_questions.append(averted_questions[idx])

            # averted_post
            averted_post_questions.append(averted_questions[idx + 1])

            # direct_pre
            direct_pre_questions.append(direct_questions[idx])

            # direct_post
            direct_post_questions.append(direct_questions[idx + 1])

            # natural_pre
            natural_pre_questions.append(natural_questions[idx])

            # natural_post
            natural_post_questions.append(natural_questions[idx + 1])

        # Scoring for each subscale (averted_pre) & Combine into a single dataframe

        # Load questionnaire data

        averted_pre_all_data_list = []
        # Create a loop that takes all files from the above list
        for idx, file in enumerate(averted_pre_questions):

            file_to_load = path2questions + file

            df = pd.read_csv(
                file_to_load,
                sep=";",
            )

            # Sum subscore of empathy
            df["Empathy SPGQ"] = df["Answer"][:7].sum()

            # Sum subscore of negative feeling
            df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

            # Sum subscore of behavioral
            df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

            # Total score of SPGQ
            subscales_spgq = [
                "Empathy SPGQ",
                "NegativeFeelings SPGQ",
                "Behavioural SPGQ",
            ]
            df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

            # Total score of copresence
            df["CoPresence Total"] = df["Answer"][7:13].sum()

            # Get first row and all columns
            df_clean = df.iloc[0:1, 4:]

            # Put into a list
            averted_pre_all_data_list.append(df_clean)

        # Combine int a single dataframe for averted pre
        df_averted_pre = pd.concat(averted_pre_all_data_list, ignore_index=True)

        # Scoring for each subscale (averted_post) & Combine into a single dataframe
        # Load questionnaire data

        averted_post_all_data_list = []

        # Create a loop that takes all files from the above list
        for idx, file in enumerate(averted_post_questions):

            file_to_load = path2questions + file

            df = pd.read_csv(
                file_to_load,
                sep=";",
            )

            # Sum subscore of empathy
            df["Empathy SPGQ"] = df["Answer"][:7].sum()

            # Sum subscore of negative feeling
            df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

            # Sum subscore of behavioral
            df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

            # Total score of SPGQ
            subscales_spgq = [
                "Empathy SPGQ",
                "NegativeFeelings SPGQ",
                "Behavioural SPGQ",
            ]
            df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

            # Total score of copresence
            df["CoPresence Total"] = df["Answer"][7:13].sum()

            # Get first row and all columns
            df_clean = df.iloc[0:1, 4:]

            # Put into a list
            averted_post_all_data_list.append(df_clean)

        # Combine int a single dataframe for averted pre
        df_averted_post = pd.concat(averted_post_all_data_list, ignore_index=True)

        # Scoring for each subscale (direct_pre) & Combine into a single dataframe
        # Load questionnaire data

        direct_pre_all_data_list = []
        # Create a loop that takes all files from the above list
        for idx, file in enumerate(direct_pre_questions):

            file_to_load = path2questions + file

            df = pd.read_csv(
                file_to_load,
                sep=";",
            )

            # Sum subscore of empathy
            df["Empathy SPGQ"] = df["Answer"][:7].sum()

            # Sum subscore of negative feeling
            df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

            # Sum subscore of behavioral
            df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

            # Total score of SPGQ
            subscales_spgq = [
                "Empathy SPGQ",
                "NegativeFeelings SPGQ",
                "Behavioural SPGQ",
            ]
            df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

            # Total score of copresence
            df["CoPresence Total"] = df["Answer"][7:13].sum()

            # Get first row and all columns
            df_clean = df.iloc[0:1, 4:]

            # Put into a list
            direct_pre_all_data_list.append(df_clean)

        # Combine int a single dataframe for averted pre
        df_direct_pre = pd.concat(direct_pre_all_data_list, ignore_index=True)

        # Scoring for each subscale (direct_post) & Combine into a single dataframe
        # Load questionnaire data

        direct_post_all_data_list = []
        # Create a loop that takes all files from the above list
        for idx, file in enumerate(direct_post_questions):

            file_to_load = path2questions + file

            df = pd.read_csv(
                file_to_load,
                sep=";",
            )

            # Sum subscore of empathy
            df["Empathy SPGQ"] = df["Answer"][:7].sum()

            # Sum subscore of negative feeling
            df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

            # Sum subscore of behavioral
            df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

            # Total score of SPGQ
            subscales_spgq = [
                "Empathy SPGQ",
                "NegativeFeelings SPGQ",
                "Behavioural SPGQ",
            ]
            df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

            # Total score of copresence
            df["CoPresence Total"] = df["Answer"][7:13].sum()

            # Get first row and all columns
            df_clean = df.iloc[0:1, 4:]

            # Put into a list
            direct_post_all_data_list.append(df_clean)

        # Combine int a single dataframe for averted pre
        df_direct_post = pd.concat(direct_post_all_data_list, ignore_index=True)

        # Scoring for each subscale (natural_pre) & Combine into a single dataframe
        # Load questionnaire data

        natural_pre_all_data_list = []
        # Create a loop that takes all files from the above list
        for idx, file in enumerate(natural_pre_questions):

            file_to_load = path2questions + file

            df = pd.read_csv(
                file_to_load,
                sep=";",
            )

            # Sum subscore of empathy
            df["Empathy SPGQ"] = df["Answer"][:7].sum()

            # Sum subscore of negative feeling
            df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

            # Sum subscore of behavioral
            df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

            # Total score of SPGQ
            subscales_spgq = [
                "Empathy SPGQ",
                "NegativeFeelings SPGQ",
                "Behavioural SPGQ",
            ]
            df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

            # Total score of copresence
            df["CoPresence Total"] = df["Answer"][7:13].sum()

            # Get first row and all columns
            df_clean = df.iloc[0:1, 4:]

            # Put into a list
            natural_pre_all_data_list.append(df_clean)

        # Combine int a single dataframe for averted pre
        df_natural_pre = pd.concat(natural_pre_all_data_list, ignore_index=True)

        # Scoring for each subscale (natural_post) & Combine into a single dataframe
        # Load questionnaire data

        natural_post_all_data_list = []
        # Create a loop that takes all files from the above list
        for idx, file in enumerate(natural_post_questions):

            file_to_load = path2questions + file

            df = pd.read_csv(
                file_to_load,
                sep=";",
            )

            # Sum subscore of empathy
            df["Empathy SPGQ"] = df["Answer"][:7].sum()

            # Sum subscore of negative feeling
            df["NegativeFeelings SPGQ"] = df["Answer"][7:13].sum()

            # Sum subscore of behavioral
            df["Behavioural SPGQ"] = df["Answer"][13:21].sum()

            # Total score of SPGQ
            subscales_spgq = [
                "Empathy SPGQ",
                "NegativeFeelings SPGQ",
                "Behavioural SPGQ",
            ]
            df["SPGQ Total"] = df[subscales_spgq].sum(axis=1)

            # Total score of copresence
            df["CoPresence Total"] = df["Answer"][7:13].sum()

            # Get first row and all columns
            df_clean = df.iloc[0:1, 4:]

            # Put into a list
            natural_post_all_data_list.append(df_clean)

        # Combine int a single dataframe for averted pre
        df_natural_post = pd.concat(natural_post_all_data_list, ignore_index=True)

        all_questionnaires = results(
            df_averted_pre,
            df_averted_post,
            df_direct_pre,
            df_direct_post,
            df_natural_pre,
            df_natural_post,
        )

        return all_questionnaires

    def diff_score_questionnaire_pre_post(
        self,
        df_averted_pre: DataFrame,
        df_averted_post: DataFrame,
        df_direct_pre: DataFrame,
        df_direct_post: DataFrame,
        df_natural_pre: DataFrame,
        df_natural_post: DataFrame,
        column_name: str,
    ):

        """
        Objective : Find a difference between post and pre of subscale of questionnnaire for each eye condition.
                    Both subscales and total score of Social Prensece Game Questionnaire (SPGQ)

        Parameters :
                    - df_averted_pre (DataFrame) : DataFrame that is resulted from scoring_questionnaire function
                    - df_averted_post (DataFrame) : DataFrame that is resulted from scoring_questionnaire function
                    - df_direct_pre (DataFrame) : DataFrame that is resulted from scoring_questionnaire function
                    - df_direct_post (DataFrame) : DataFrame that is resulted from scoring_questionnaire function
                    - df_natural_pre (DataFrame) : DataFrame that is resulted from scoring_questionnaire function
                    - df_natural_post (DataFrame) : DataFrame that is resulted from scoring_questionnaire function
                    - column_name (str) : Choose one of the options below
                                        - "Empathy SPGQ"
                                        - "NegativeFeelings SPGQ"
                                        - "Behavioural SPGQ"
                                        - "SPGQ Total"
                                        - "CoPresence Total"

        Outputs    :
                    There are 3 outputs which are the result of substraction post and pre of
                       (subscale / SPGQ total) questionnnaire :
                    - substracted_averted
                    - substracted_direct
                    - substracted_natural

                    Note : Related to questionnaires
                            There are 2 questionnaires here that we use in the experiment :
                            * Social Presence in Gaming Questionnaire (SPGQ), which consists of 3 subscales (Higher score, Higher Social Presence)
                                * Psychological involvement - Empathy
                                * Psychological involvement - Negative feelings
                                * Psychological involvement - Behavioral engagement
                            * Co-Presence questionnaire (REMEMBER : HIGER score, indicates LESS CoPresence !!!)
                            * See here for details https://docs.google.com/document/d/118ZIYY5o2bhJ6LF0fYcxDA8iinaLcn1EZ5V77zt_AeQ/edit#
        """

        df_averted_pre_list = df_averted_pre[column_name].tolist()
        df_averted_post_list = df_averted_post[column_name].tolist()
        df_direct_pre_list = df_direct_pre[column_name].tolist()
        df_direct_post_list = df_direct_post[column_name].tolist()
        df_natural_pre_list = df_natural_pre[column_name].tolist()
        df_natural_post_list = df_natural_post[column_name].tolist()

        df_averted_pre_combined = []
        df_direct_pre_combined = []
        df_natural_pre_combined = []

        df_averted_post_combined = []
        df_direct_post_combined = []
        df_natural_post_combined = []

        begin = 0
        end = len(df_averted_pre_list)
        step = 2
        for idx in range(begin, end, step):
            # Pre conditions
            df_averted_pre_combined.append(
                (df_averted_pre_list[idx] + df_averted_pre_list[idx + 1]) / 2
            )
            df_direct_pre_combined.append(
                (df_direct_pre_list[idx] + df_direct_pre_list[idx + 1]) / 2
            )
            df_natural_pre_combined.append(
                (df_natural_pre_list[idx] + df_natural_pre_list[idx + 1]) / 2
            )

            # Post conditions
            df_averted_post_combined.append(
                (df_averted_post_list[idx] + df_averted_post_list[idx + 1]) / 2
            )
            df_direct_post_combined.append(
                (df_direct_post_list[idx] + df_direct_post_list[idx + 1]) / 2
            )
            df_natural_post_combined.append(
                (df_natural_post_list[idx] + df_natural_post_list[idx + 1]) / 2
            )

        # Substract post and pre score of subscale / SPGQ Total score. Depending on the input of parameter
        substracted_averted = np.abs(
            [
                averted_post - averted_pre
                for averted_post, averted_pre in zip(
                    df_averted_post_combined, df_averted_pre_combined
                )
            ]
        )
        substracted_direct = np.abs(
            [
                direct_post - direct_pre
                for direct_post, direct_pre in zip(
                    df_direct_post_combined, df_direct_pre_combined
                )
            ]
        )
        substracted_natural = np.abs(
            [
                natural_post - natural_pre
                for natural_post, natural_pre in zip(
                    df_natural_post_combined, df_natural_pre_combined
                )
            ]
        )

        return substracted_averted, substracted_direct, substracted_natural
