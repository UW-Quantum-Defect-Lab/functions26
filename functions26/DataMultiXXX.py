# 2020-09-14
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# based on ideas and code of Christian Zimmermann
# Added/Edited by Chris on 2020-09-21

import warnings
from .DataXXX import DataSIF, DataOP, DataT1
from .DataDictXXX import DataDictFilenameInfo
from .DataFrame26 import DataFrame26


class DataMultiXXX:

    def __init__(self, filename_list, folder_name):
        self.filename_list = filename_list
        if not self.filename_list:
            raise RuntimeError('Filename list is empty')
        self.size = len(self.filename_list)
        self.folder_name = folder_name
        self.data_object_list = self.get_data_object_list()

        self.multi_file_info = DataDictFilenameInfo()
        self.multi_data = DataFrame26()
        self.get_multi_file_info()

    def get_data_object_list(self):
        warnings.warn('Define your own get_data_list() function')
        return []

    def get_multi_file_info(self):

        self.initialize_multi_file_info()

        for i in range(self.size):  # for object in object list
            data_object = self.data_object_list[i]
            for key in data_object.file_info:  # iterate through object keys
                self.multi_file_info[key][i] = data_object.file_info[key]

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


class DataMultiSIF(DataMultiXXX):

    def __init__(self, file_name_list, second_order=False, wavelength_offset=0, background_per_cycle=300,
                 folder_name='.', from_video=True):
        self.second_order = second_order
        self.wavelength_offset = wavelength_offset
        self.background_per_cycle = background_per_cycle
        self.from_video = from_video
        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self):
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataSIF(file_name=file_name,
                                            second_order=self.second_order,
                                            wavelength_offset=self.wavelength_offset,
                                            background_per_cycle=self.background_per_cycle,
                                            folder_name=self.folder_name,
                                            from_video=self.from_video))

        return data_object_list

    def get_integrated_pl(self, label, unit_y, subtract_bg=False, range='entire', unit_x='nm', other_variables=None):
        # Sort self.data_object_list
        if isinstance(other_variables, str):
            other_variables = [other_variables]
        elif other_variables is None:
            other_variables = []

        dictionary_keys = ['x_' + unit_x, 'PL']
        for variable in other_variables:
            if variable in self.multi_file_info:
                dictionary_keys += other_variables

        try:
            sorted_data_object_list = sorted(self.data_object_list,
                                             key=lambda data_object: float(data_object.file_info[label]), reverse=True)
        except (ValueError, TypeError) as e:
            sorted_data_object_list = self.data_object_list
            warnings.warn('In get_intergrated_pl -> Data Object List could not be sorted')

        integrated_pl = {key: [] for key in dictionary_keys}
        for n, data_object in enumerate(sorted_data_object_list):
            integrated_pl['x_' + unit_x].append(float(data_object.file_info[label]))
            if range == 'entire':
                y = data_object.integrate_counts(unit_y, subtract_bg)
            else:
                y = data_object.integrate_in_region(range[0], range[1], unit_x, unit_y, subtract_bg)
            integrated_pl['PL'].append(y)
            for key in dictionary_keys[2:]:
                integrated_pl[key].append(data_object.file_info[key])

        integrated_pl['DataSIF'] = sorted_data_object_list
        integrated_pl_df = DataFrame26(data=integrated_pl)
        return integrated_pl_df


class DataMultiOP(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.'):

        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self):
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataOP(file_name=file_name, folder_name=self.folder_name))

        return data_object_list


class DataMultiT1(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.'):

        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self):
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataT1(file_name=file_name, folder_name=self.folder_name))

        return data_object_list
