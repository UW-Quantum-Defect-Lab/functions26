import numpy as np
import os

from os import path
from pandas import DataFrame

from .DataFrame26 import DataFrame26
from .DataMultiXXX import DataMultiPower, DataMultiSPCM
from .DataXXX import DataPower, DataSPCM, DataWavelength
from .FilenameManager import FilenameManager
from .FittingManager import linear_sine_fit
from .constants import n_air, conversion_factor_nm_to_ev
from .filing.QDLFiling import MultiQDLF
from .units import unit_families

power_oscillation_frequency = {'nm': 1 / 0.00615929629766, 'eV': 0.00615929629766}
power_oscillation_amplitude = 0.08
power_oscillation_percentage = 0.58


def get_oscillation(x_data, y_counts_per_power, x_units, max_oscillation_amplitude_to_signal_ratio=0.3):
    if x_units not in ['nm', 'eV']:
        raise ValueError('x_units must be either nm or eV.')

    x_data = np.array(x_data)
    y_data = np.array(y_counts_per_power)
    signal_height = max(y_counts_per_power) - min(y_counts_per_power)
    oscilalation_max_height = signal_height * max_oscillation_amplitude_to_signal_ratio + min(y_counts_per_power)

    only_bg_x_data = np.array(x_data[y_counts_per_power < oscilalation_max_height])
    only_bg_y_data = np.array(y_data[y_counts_per_power < oscilalation_max_height])

    fitmng = linear_sine_fit(only_bg_x_data, only_bg_y_data, model_guess_index_regions=None,
                             input_parameters=DataFrame({'names': ['frequency'],
                                                         'initial_values': [power_oscillation_frequency[x_units]],
                                                         'is_fixed': [True]}))

    correction_parameters = fitmng.fit_result.params
    correction_parameters['amplitude'].value = power_oscillation_amplitude
    # correction_parameters['intercept'].value = power_oscillation_percentage
    # correction_parameters['slope'].value = 0

    sine_bg_data = fitmng.fit_result.model.eval_components(params=correction_parameters, x=x_data)['sine'] \
                   + power_oscillation_percentage

    return sine_bg_data, correction_parameters['shift']


class DataAutoStationaryPLE:
    spacer = '_'
    measurement_types_list = ['power', 'spcm']
    measurement_types_classes = {'power': DataMultiPower, 'spcm': DataMultiSPCM}

    def __init__(self, folder=None, filenames=None, varying_variable='Lsr: Wavelength (nm)', second_order=True,
                 max_oscillation_amplitude_to_signal_ratio=0.3, refractive_index=n_air, want_oscillation=True):
        """Chose either a folder or give a list of filenames"""
        self._set_filenames_and_manager(folder, filenames)

        self.varying_variable = varying_variable
        self.x_unit = self.varying_variable.split('(')[-1].split(')')[0]

        self.second_order = second_order
        self.refractive_index = refractive_index
        self.max_oscillation_amplitude_to_signal_ratio = max_oscillation_amplitude_to_signal_ratio
        self.want_oscillation = want_oscillation

        self._get_measurement_filenames()
        self._get_files()
        self.__post_init__()

    def __post_init__(self):
        self._set_data()

    def _set_filenames_and_manager(self, folder, filenames):
        if folder is not None:
            self.filenames = [path.join(folder, f) for f in os.listdir(folder)
                              if path.isfile(path.join(folder, f)) and f.endswith('.qdlf')]
            self.filenames.sort()
        else:
            self.filenames = filenames
        self.fnm = FilenameManager(self.filenames)

    def _get_measurement_filenames(self):
        self.measurement_filenames = {measurement_type: [] for measurement_type in self.measurement_types_list}

        for filename in self.filenames:
            file_info = self.fnm._get_file_info(filename)
            for measurement_type in self.measurement_types_list:
                if file_info['Measurement Type'] == measurement_type:
                    self.measurement_filenames[measurement_type].append(filename)

    def _get_files(self):
        self.multi_objects = {}
        for measurement_type in self.measurement_filenames:
            filenames = self.measurement_filenames[measurement_type]
            multi_object = self.measurement_types_classes[measurement_type](filenames)
            self.multi_objects[measurement_type] = multi_object

    def _set_data(self):
        self.data = DataFrame26([], unit_families, self.spacer)
        self._set_x_data()
        self._set_y_data()
        self.data.sort_values(by='x_' + self.x_unit, ignore_index=True)

    def _set_x_data(self):
        self._set_original_x_data()
        self._set_other_x_data()

    def _set_y_data(self):
        self._set_original_y_data()
        self._set_power_corrected_y_data()
        if self.want_oscillation:
            self._get_oscillation()
            self._set_oscillation_corrected_y_data()

    def _set_original_x_data(self):
        x_data = []
        for filename in self.filenames:
            file_info = self.fnm._get_file_info(filename)
            x_element = file_info[self.varying_variable]
            if file_info['Measurement Type'] == self.measurement_types_list[0]:  # to not double count
                x_data.append(x_element)

        if self.second_order is True:
            self.data['x_' + self.x_unit] = np.array(x_data)/2
        else:
            self.data['x_' + self.x_unit] = np.array(x_data)

    def _set_other_x_data(self):
        if self.x_unit == 'nm':
            other_x_unit = 'eV'
        else:
            other_x_unit = 'nm'
        self.data['x_' + other_x_unit] = conversion_factor_nm_to_ev/(self.data['x_' + self.x_unit]
                                                                     * self.refractive_index)

    def _set_original_y_data(self):
        for measurement_type in self.measurement_types_list:
            averages = self.multi_objects[measurement_type].averages
            stdevs = self.multi_objects[measurement_type].stdevs
            for key in averages:
                if key.startswith('y'):
                    unit = ''.join(key.split('y')[1:])
                    self.data[measurement_type + '_mean' + unit] = averages[key]
                    self.data[measurement_type + '_stdev' + unit] = stdevs[key]

    def _set_power_corrected_y_data(self):
        if 'power_mean_uW' in self.data.keys() and 'spcm_mean_counts' in self.data.keys():
            self.data['y_mean_counts_per_uW'] = self.data['spcm_mean_counts'] / self.data['power_mean_uW']

    def _get_oscillation(self):
        if 'y_mean_counts_per_uW' in self.data.keys():
            x_data = np.array(self.data['x_nm'])
            if self.second_order:
                x_data = x_data*2
            y_data = np.array(self.data['y_mean_counts_per_uW'])
            self.data['sine_correction'], self.sine_correction_phase = \
                get_oscillation(x_data, y_data, 'nm', self.max_oscillation_amplitude_to_signal_ratio)

    def _set_oscillation_corrected_y_data(self):
        if 'sine_correction' in self.data.keys():
            y_data = np.array(self.data['y_mean_counts_per_uW'])
            power_data = self.data['power_mean_uW']
            sine_correction_data = np.array(self.data['sine_correction'])
            self.data['y_osc_corrected_mean_counts_per_uW'] = y_data/sine_correction_data
            self.data['power_osc_corrected_mean_uW'] = power_data*sine_correction_data


class DataAutoContinuousPLE:
    spacer = '_'
    measurement_types_list = ['power', 'spcm', 'wavelength']
    measurement_types_classes = {'power': DataPower, 'spcm': DataSPCM, 'wavelength': DataWavelength}
    allowed_file_extensions = ['mqdlf']
    default_keys = []
    allowed_units = unit_families
    qdlf_datatype = 'ContinuousWavelengthScanPLE'

    def __init__(self, filename, folder='./', second_order=True,
                 max_oscillation_amplitude_to_signal_ratio=0.3, refractive_index=n_air):
        """Give an mqdlf filename"""
        self.filename = path.join(folder, filename)
        self.max_oscillation_amplitude_to_signal_ratio = max_oscillation_amplitude_to_signal_ratio
        self.refractive_index = refractive_index

        self.second_order = second_order

        self._get_files()
        self.__post_init__()

    def _get_files(self):
        self.multi_data_manager = MultiQDLF.load(self.filename)
        self.objects = {}
        for data_manager in self.multi_data_manager.data_managers:
            if data_manager.datatype in self.measurement_types_list:
                self.objects[data_manager.datatype] = self.measurement_types_classes[
                    data_manager.datatype]('.in_code_data_manager', data_manager=data_manager)

    def __post_init__(self):
        self._set_all_data()

    def _set_all_data(self):
        self.data = DataFrame26(self.default_keys, self.allowed_units, self.spacer)
        self._set_x_data()
        self._set_y_data()

    def _set_x_data(self):
        self._get_two_point_wavelength_data()

    def _get_two_point_wavelength_data(self):
        initial_wl = self.objects['wavelength'].data['y_nm'][0]
        final_wl = np.array(self.objects['wavelength'].data['y_nm'])[-2]
        max_object_data_length = max([len(self.objects[key].data['x_second']) for key in self.objects.keys()])

        # The following needs to be more intricate and include time
        if self.second_order is True:
            self.data['x_nm'] = np.linspace(initial_wl, final_wl, max_object_data_length) / 2
        else:
            self.data['x_nm'] = np.linspace(initial_wl, final_wl, max_object_data_length)
        self.data['x_eV'] = conversion_factor_nm_to_ev / (self.data['x_nm'] * self.refractive_index)

    def _set_y_data(self):
        self.data['y_counts'] = self.objects['spcm'].data['y_counts']


