# Last modified: 2022-02-24
# by Vasilis Niaouris

import os
import warnings
from typing import Union, List

from .DataDictXXX import DataDictFilenameInfo
from .DataFrame26 import DataFrame26


class FilenameManager:
    """
    FilenameManager is a class that allows the user to retrieve the filename information of multiple files.
    When we say filename information, we refer to the specific language we use for to efficiently store the conditions
    under which the data were taken.

    Attributes
    ----------
    filenames: List[str]
        The list of filenames the rest of information this class is holding.
    size: int
        The size/length of the filenames list.
    multi_file_info: DataDictFilenameInfo
        A dictionary of all the file information. If the information is the same for all files, then the value of the
        information key will be a single value, otherwise it will be a list of values with length equal to the length of
        the filename list.
    multi_data: DataFrame26
        A dataframe containing all the information that is different among the list of files.
    """
    def __init__(self, filenames: List[str] = None):
        """
        Parameters
        ----------
        filenames: List[str] or str
            The filename or list of filenames of interest.
        """
        if isinstance(filenames, str):
            filenames = [filenames]

        self.filenames = filenames
        self.size = len(filenames)
        self.multi_file_info = DataDictFilenameInfo()
        self.multi_data = DataFrame26()

        self._get_multi_file_info()

    @staticmethod
    def _get_file_info(file_name) -> DataDictFilenameInfo:
        """
        A method that gets the information from single filename.

        Parameters
        ----------
        file_name: str
            Filename whose information will be extracted.

        Returns
        -------
        DataDictFilenameInfo
            A dictionary that contains all the relative information in the file name.
        """
        # get filename without folder and file extension
        file_info_raw = '.'.join(file_name.split('.')[:-1])
        if '/' in file_name:
            file_info_raw = file_info_raw.split('/')[-1]

        file_info_raw_components = file_info_raw.split('_')  # All file info are separated by '_'
        file_info = DataDictFilenameInfo()
        file_info.get_info(file_info_raw_components)  # retrieve info from file
        return file_info

    def get_file_info_by_name(self, file_name: str) -> DataDictFilenameInfo:
        """
        Parameters
        ----------
        file_name: str
            The file name you want the information for.
        Returns
        -------
        DataDictFilenameInfo
            A dictionary that contains all the relative information in the file name.
        """
        return self._get_file_info(file_name)

    def get_file_info_by_index(self, index: int):
        """
        Parameters
        ----------
        index: int
            The index of the file name in the filenames list that you want to use.
        Returns
        -------
        DataDictFilenameInfo
            A dictionary that contains all the relative information in the file name.
        """
        return self._get_file_info(self.filenames[index])

    def _get_multi_file_info(self):
        """
        A method to retrieve all the information of the multi file information.
        It takes each individual file info, and at the end, for every key in the file info, if for all file names the
        values are the same, the multi file info dict will hold only this common value, otherwise, it will hold a list
        of values.
        """

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

    def initialize_multi_file_info(self) -> bool:
        """
        Initialize multi_file_info values to none lists instead of single none.
        In multi_file_info every value of a specific key corresponds to a list of values.
        When parsing the lists for each dictionary value, we would run to an error if the list was replaced with None.
        Instead, we make a list of None, so the list parsing will not raise any errors.
        """
        for key in self.multi_file_info:
            self.multi_file_info[key] = [None]*self.size

        return True


class FileNumberManager(FilenameManager):
    """
    FileNumberManager is a FilenameManager subclass that allows the user to input a list of file numbers, a list
    or a single file type (e.g. 'sif', 'csv', 'qdlf' etc.) and a folder name (if needed) and retrieve all of the file
    information and filenames of all the files with the given number and file type.

    Attributes
    ----------
    file_no_list:
        A list of all the potential file numbers of interest (e.g. 1, 2, 45, 1000 etc.)
    file_types:
        A list of all the potential file types of interest (e.g. 'sif', 'csv', 'qdlf' etc.)
    folder_name:
        The folder name in which the files are located in.
    filenumbers:
        The list of all the retrieved filenumbers (contains duplicates if there are two
         files with the same filename number)

    Examples
    --------
    Assume files of type 'sif' in a folder 'data/sif_data' with filenumbers [1, 2, 5].
    >>> fnm = FileNumberManager([1, 2, 5], 'sif', folder_name='data/sif_data')
    >>> filenames = fnm.filenames
    >>> multi_file_info = fnm.multi_file_info
    >>> multi_data = fnm.multi_data
    These are the most commonly used attributes of FileNumberManager.

    You can also use this class to extract individual DataDictFilenmaeInfo dictionaries:
    >>>

    >>>

    """

    def __init__(self, file_no_list: Union[List[int], int], file_types: Union[List[str], str], folder_name: str = '.'):
        """
        Parameters
        ----------
        file_no_list: List[int] or int
            A file number or list of all the potential file numbers of interest (e.g. 1, 2, 45, 1000 etc.)
        file_types: List[str] or str
            A file type or list of all the potential file types of interest (e.g. 'sif', 'csv', 'qdlf' etc.)
        folder_name:
        The folder name in which the files are located in.
        """

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
                warnings.warn(str(file_no) + ' is not a numeric.')

        filenames = self._get_filenames()
        super().__init__(filenames)
        self.filenumbers = self.multi_file_info['File Number']

    def _get_all_file_names_with_specific_file_types(self) -> List[str]:
        """
        Finds all the files in the folder with the given file type.

        Returns
        -------
        List[str]
            A list of all the files in given folder with the given file type.
        """
        filenames = []
        for file_name in os.listdir(self.folder_name):
            for file_type in self.file_types:
                if file_name.endswith(file_type):
                    filenames.append(file_name)
        return filenames

    def _get_filenames(self) -> List[str]:
        """
        Finds all the files in the folder with the given file type and the given file numbers.

        Returns
        -------
        List[str]
            A list of all the files in given folder with the given file type and the given filenumbers.
        """

        # gets all the filenames with the right file type.
        all_file_names_with_specific_file_types = self._get_all_file_names_with_specific_file_types()
        filenames = []

        # gets all the filenames with the right file number.
        for filename in all_file_names_with_specific_file_types:

            if filename.split('_')[0].isnumeric():
                if int(filename.split('_')[0]) in self.file_no_list:
                    filenames.append(filename)
        filenames.sort()

        # Adding the folder in each filename in the filenames list
        for i in range(len(filenames)):
            filenames[i] = os.path.join(self.folder_name, filenames[i])


        return filenames
