# 2020-09-14
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# based on ideas and code of Christian Zimmermann
# Added/Edited by Chris on 2020-09-21

import numpy as np
import pandas as pd
import scipy.optimize as spo
import warnings

from typing import List, Union
from .constants import n_air
from .constants import conversion_factor_nm_to_ev
from .DataFrame26 import DataFrame26
from .DataDictXXX import DataDictFilenameInfo
from .DataXXX import DataSIF, DataOP, DataOP2LaserDelay, DataT1, DataRFSpectrum, DataSPCMCounter
from .Plot2D import two_dimensional_plot


def line(x, a, b):
    return a * x + b


def exponential(x, a, b, x0):
    return a * np.exp(b * (x - x0))


def guess_initial_parameters(x, y, function='linear'):
    index_min = x.idxmin()
    index_max = x.idxmax()
    x0 = x.loc[index_min]
    x1 = x.loc[index_max]
    y0 = y.loc[index_min]
    y1 = y.loc[index_max]
    if function == 'linear':
        slope = (y1 - y0) / (x1 - x0)
        intersect = y1 - slope * x1
        p0 = [slope, intersect]
    elif function == 'exponential':
        exponent = (np.log(y1) - np.log(y0)) / (x1 - x0)
        prefactor = y0
        shift = x0
        p0 = [prefactor, exponent, shift]

    return p0


class DataMultiXXX:

    def __init__(self, filename_list=None, folder_name='.', qdlf_datatype=None):
        if filename_list is None:
            filename_list = []
        self.filename_list = filename_list
        self.size = len(self.filename_list)
        self.folder_name = folder_name
        self.qdlf_datatype = qdlf_datatype

        if not self.filename_list:
            raise ValueError('Filename list is empty')
        else:
            self.data_object_list = self.get_data_object_list()

            self.multi_file_info = DataDictFilenameInfo()
            self.multi_data = DataFrame26(spacer=self.data_object_list[0].spacer)
            self.get_multi_file_info()

    def __iter__(self):
        return iter(self.data_object_list)

    def get_data_object_list(self) -> List[Union[DataSIF, DataOP, DataOP2LaserDelay, DataT1, DataRFSpectrum,
                                                 DataSPCMCounter]]:
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
            self.multi_file_info[key] = [None] * self.size

        return True


class DataMultiSIF(DataMultiXXX):

    def __init__(self, file_name_list, second_order=False, wavelength_offset=0, refractive_index=n_air,
                 background_per_cycle=300, folder_name='.', from_video=True):
        self.second_order = second_order
        self.wavelength_offset = wavelength_offset
        self.refractive_index = refractive_index
        self.background_per_cycle = background_per_cycle
        self.from_video = from_video
        super().__init__(file_name_list, folder_name)
        self.baseline = {}
        self.heatmap = None

    def get_data_object_list(self) -> List[DataSIF]:
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataSIF(file_name=file_name,
                                            second_order=self.second_order,
                                            wavelength_offset=self.wavelength_offset,
                                            refractive_index=self.refractive_index,
                                            background_per_cycle=self.background_per_cycle,
                                            folder_name=self.folder_name,
                                            from_video=self.from_video))

        return data_object_list

    def get_integrated_pl(self, label, unit_y, subtract_bg=False, range='entire', unit_x='nm', other_variables=None,
                          baseline=None):
        # Sort self.data_object_list
        if isinstance(other_variables, str):
            other_variables = [other_variables]
        elif other_variables is None:
            other_variables = []

        dictionary_keys = [label, 'PL']
        for variable in other_variables:
            if variable in self.multi_file_info:
                dictionary_keys += [variable]

        try:
            sorted_data_object_list = sorted(self.data_object_list,
                                             key=lambda data_object: float(data_object.file_info[label]), reverse=True)
        except (ValueError, TypeError) as e:
            sorted_data_object_list = self.data_object_list
            warnings.warn('In get_intergrated_pl -> Data Object List could not be sorted')

        if baseline is not None:
            self.baseline = {}

        integrated_pl = {key: [] for key in dictionary_keys}
        for n, data_object in enumerate(sorted_data_object_list):
            label_value = float(data_object.file_info[label])
            integrated_pl[label].append(label_value)
            if range == 'entire':
                y = data_object.integrate_counts(unit_y, subtract_bg)
            else:
                y = data_object.integrate_in_region(range[0], range[1], unit_x, unit_y, subtract_bg)
                if baseline == 'linear':
                    self.baseline[label_value] = {}
                    index_left = np.abs(data_object.data['x_{0}'.format(unit_x)] - range[0]).idxmin()
                    index_right = np.abs(data_object.data['x_{0}'.format(unit_x)] - range[1]).idxmin()
                    sub_spectrum_for_fitting_left = data_object.data.loc[index_left - 5: index_left + 5]
                    sub_spectrum_for_fitting_right = data_object.data.loc[index_right - 5: index_right + 5]
                    sub_spectrum_for_fitting = pd.concat([sub_spectrum_for_fitting_left,
                                                          sub_spectrum_for_fitting_right], ignore_index=True)
                    p0 = guess_initial_parameters(sub_spectrum_for_fitting['x_{0}'.format(unit_x)],
                                                  sub_spectrum_for_fitting['y_{0}'.format(unit_y)], 'linear')
                    parameters, covariance = spo.curve_fit(line, sub_spectrum_for_fitting['x_{0}'.format(unit_x)],
                                                           sub_spectrum_for_fitting['y_{0}'.format(unit_y)], p0=p0)
                    self.baseline[label_value]['slope_initial'] = p0[0]
                    self.baseline[label_value]['intersect_initial'] = p0[1]
                    self.baseline[label_value]['slope'] = parameters[0]
                    self.baseline[label_value]['intersect'] = parameters[1]
                if baseline == 'linear':
                    sub_spectrum = data_object.data.loc[(data_object.data['x_{0}'.format(unit_x)] >= range[0]) & (
                            data_object.data['x_{0}'.format(unit_x)] <= range[1])]
                    y = y - (line(sub_spectrum['x_{0}'.format(unit_x)], self.baseline[label_value]['slope'],
                                  self.baseline[label_value]['intersect']) - data_object.infoSIF[
                                 'background_{0}'.format(unit_y)]).sum()
            integrated_pl['PL'].append(y)
            for key in dictionary_keys[2:]:
                integrated_pl[key].append(data_object.file_info[key])

        integrated_pl['DataSIF'] = sorted_data_object_list
        integrated_pl_df = DataFrame26(qdlf_datatype='PLE', data=integrated_pl)

        # Add Photon Energy
        if label == 'Lsr: Wavelength (nm)':
            integrated_pl_df['Lsr: Energy (eV)'] = conversion_factor_nm_to_ev / (
                        integrated_pl_df['Lsr: Wavelength (nm)']
                        * self.refractive_index / 2)
        if label == 'Ls2: Wavelength (nm)':
            integrated_pl_df['Ls2: Energy (eV)'] = conversion_factor_nm_to_ev / (
                        integrated_pl_df['Ls2: Wavelength (nm)']
                        * self.refractive_index / 2)
        return integrated_pl_df

    def add_heatmap(self,
                    data_identifier='y_nobg_counts_per_second',
                    x_axis_identifier='x_eV',
                    y_axis_identifier='Lsr: Wavelength (nm)',
                    axes_limits='Auto',
                    scale='Auto',
                    color_bar=True,
                    shading='auto',
                    plot_style=None):
        try:
            sorted_data_object_list = sorted(self.data_object_list,
                                             key=lambda data_object: float(data_object.file_info[y_axis_identifier]),
                                             reverse=False)
        except (ValueError, TypeError) as e:
            sorted_data_object_list = self.data_object_list
            warnings.warn('In get_intergrated_pl -> Data Object List could not be sorted')

        x_axis = []
        y_axis = []
        data = []

        if x_axis_identifier == 'x_eV':
            x_axis = self.data_object_list[0].data[x_axis_identifier]
        elif x_axis_identifier == 'x_nm':
            x_axis = self.data_object_list[0].data[x_axis_identifier]
        x_axis.reset_index(inplace=True, drop=True)

        for data_object in sorted_data_object_list:
            if x_axis_identifier == 'x_eV':
                data.append(data_object.data[data_identifier])
            elif x_axis_identifier == 'x_nm':
                data.append(data_object.data[data_identifier])
            y_axis.append(data_object.file_info[y_axis_identifier])

        x_axis_label, y_axis_label, color_bar_label = self.get_heatmap_labels(x_axis_identifier, y_axis_identifier,
                                                                              data_identifier)

        self.heatmap = two_dimensional_plot(data, x_axis, y_axis,
                                            x_axis_label=x_axis_label,
                                            y_axis_label=y_axis_label,
                                            axes_limits=axes_limits,
                                            scale=scale,
                                            color_bar=color_bar,
                                            color_bar_label=color_bar_label,
                                            shading=shading,
                                            plot_style=plot_style)

        return True

    @staticmethod
    def get_heatmap_labels(x_axis_identifier, y_axis_identifier, data_identifier):
        x_axis_label = ''
        if x_axis_identifier == 'x_eV':
            x_axis_label = 'Photon Energy (eV)'
        elif x_axis_identifier == 'x_nm':
            x_axis_label = 'Wavelength (nm)'

        color_bar_label = ''
        if data_identifier == 'y_nobg_counts_per_second':
            color_bar_label = 'PL-Intensity (counts/second)'
        elif data_identifier == 'y_counts_per_second':
            color_bar_label = 'PL-Intensity (counts/second), no background-correction'

        y_axis_label = ''
        if y_axis_identifier == 'Lsr: Wavelength (nm)':
            y_axis_label = 'Excitation Wavelength (nm)'
        elif y_axis_identifier == 'Temperature (K)':
            y_axis_label = 'Temperature (K)'
        elif y_axis_identifier == 'Lsr: Power (nW)':
            y_axis_label = 'Power (nW)'
        elif y_axis_identifier == 'Magnetic Field (T)':
            y_axis_label = 'Magnetic Field (T)'

        return x_axis_label, y_axis_label, color_bar_label


class DataMultiOP(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.', run_simple_fit_functions=True):
        self.run_simple_fit_functions = run_simple_fit_functions
        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self) -> List[DataOP]:
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataOP(file_name=file_name, folder_name=self.folder_name,
                                           run_simple_fit_functions=self.run_simple_fit_functions))

        return data_object_list


class DataMultiOP2LaserDelay(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.', run_simple_fit_functions=True):
        self.run_simple_fit_functions = run_simple_fit_functions
        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self) -> List[DataOP2LaserDelay]:
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataOP2LaserDelay(file_name=file_name, folder_name=self.folder_name,
                                                      run_simple_fit_functions=self.run_simple_fit_functions))

        return data_object_list


class DataMultiT1(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.'):
        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self) -> List[DataT1]:
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataT1(file_name=file_name, folder_name=self.folder_name))

        return data_object_list


class DataMultiRFSpectrum(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.'):
        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self) -> List[DataRFSpectrum]:
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataRFSpectrum(file_name=file_name, folder_name=self.folder_name))

        return data_object_list


class DataMultiSPCMCounter(DataMultiXXX):

    def __init__(self, file_name_list, folder_name='.'):
        super().__init__(file_name_list, folder_name)

    def get_data_object_list(self) -> List[DataSPCMCounter]:
        data_object_list = []
        for file_name in self.filename_list:
            data_object_list.append(DataSPCMCounter(file_name=file_name, folder_name=self.folder_name))

        return data_object_list
