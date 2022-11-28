"""
    This is a collection of abstract classes and functions (interfaces)
    that are related to EEG data processing.

"""

from abc import ABC, abstractmethod


# Pre-process class of exp2 redesign
class pre_eeg_exp2_redesign(ABC):
    @abstractmethod
    def extract_baseline_eeg_data(self):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def extract_experimental_eeg_data(self):
        raise NotImplementedError("Please Implement this method")


# Processing (Analysis) class of exp2 redesign
class analysis_eeg_exp2_redesign(ABC):
    @abstractmethod
    def analysis_hyperscanning(self):
        raise NotImplementedError("Please Implement this method")
