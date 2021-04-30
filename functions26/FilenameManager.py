
import os
import warnings

from .DataDictXXX import DataDictFilenameInfo
from .DataFrame26 import DataFrame26


class FilenameManager:

    def __init__(self, filenames=None):
        if isinstance(filenames, str):
            filenames = [filenames]

        self.filenames = filenames
        self.size = len(filenames)
        self.multi_file_info = DataDictFilenameInfo()
        self.multi_data = DataFrame26()

        self._get_multi_file_info()

    @staticmethod
    def _get_file_info(file_name):

        # get filename without folder and file extension
        file_info_raw = '.'.join(file_name.split('.')[:-1])
        if '/' in file_name:
            file_info_raw = file_info_raw.split('/')[-1]

        file_info_raw_components = file_info_raw.split('_')  # All file info are separated by '_'
        file_info = DataDictFilenameInfo()
        file_info.get_info(file_info_raw_components)  # retrieve info from file
        return file_info

    def _get_multi_file_info(self):

        self.initialize_multi_file_info()

        for i in range(self.size):  # for object in object list
            file_info = self._get_file_info(self.filenames[i])
            for key in file_info:  # iterate through object keys
                self.multi_file_info[key][i] = file_info[key]

        # # compress identical information to 1 value instead of a list
        for key in self.multi_file_info:
            value_list = self.multi_file_info[key]
            if all(value == value_list[0] for value in value_list):
                self.multi_file_info[key] = value_list[0]
            else:
                self.multi_file_info[key] = value_list  # not redundant, it corrects attribute assignment
                if key not in self.multi_file_info.fai_head_keys_dict.keys():
                    self.multi_data[key] = value_list

    def initialize_multi_file_info(self):

        # Initialize multi_file_info values to none lists instead of single none
        for key in self.multi_file_info:
            self.multi_file_info[key] = [None]*self.size

        return True


class FileNumberManager(FilenameManager):

    def __init__(self, file_no_list, file_types, folder_name='.'):

        if isinstance(file_no_list, int):
            file_no_list = [file_no_list]
        if isinstance(file_types, str):
            file_types = [file_types]

        self.file_no_list = []
        self.file_types = file_types
        self.folder_name = folder_name

        for file_no in file_no_list:
            try:
                self.file_no_list.append(int(file_no))
            except ValueError:
                warnings.warn(str(file_no) + ' is not a numeric')

        filenames = self._get_filenames()
        super().__init__(filenames)

    def _get_all_file_names_with_specific_file_types(self):

        filenames = []
        for file_name in os.listdir(self.folder_name):
            for file_type in self.file_types:
                if file_name.endswith(file_type):
                    filenames.append(file_name)
        return filenames

    def _get_filenames(self):

        all_file_names_with_specific_file_types = self._get_all_file_names_with_specific_file_types()
        filenames = []

        for filename in all_file_names_with_specific_file_types:
            if int(filename.split('_')[0]) in self.file_no_list:
                filenames.append(filename)

        filenames.sort()
        return filenames
