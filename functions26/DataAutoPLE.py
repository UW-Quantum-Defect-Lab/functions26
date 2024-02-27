from typing import List, Union

import numpy as np
import os

from os import path
from pandas import DataFrame
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit, minimize

from .DataFrame26 import DataFrame26
from .DataMultiXXX import DataMultiPower, DataMultiSPCM
from .DataXXX import DataPower, DataSPCM, DataWavelength
from .FilenameManager import FilenameManager
from .FittingManager import linear_sine_fit
from .constants import n_air, conversion_factor_nm_to_ev
from .filing.QDLFiling import MultiQDLF
from .units import unit_families

power_oscillation_frequency = {'nm': 1 / 0.00615929629766, 'eV': 1 / 0.05729E-3}
power_oscillation_amplitude = 0.08
power_oscillation_percentage = 0.58


def line(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def get_oscillation(x_data, y_counts_per_power, x_units, max_oscillation_amplitude_to_signal_ratio=0.3,
                    exclude_points_around_peak=None):
    if x_units not in ['nm', 'eV']:
        raise ValueError('x_units must be either nm or eV.')

    x_data = np.array(x_data)
    y_data = np.array(y_counts_per_power)
    signal_height = max(y_counts_per_power) - min(y_counts_per_power)
    oscilalation_max_height = signal_height * max_oscillation_amplitude_to_signal_ratio + min(y_counts_per_power)

    only_bg_x_data = np.array(x_data[y_counts_per_power < oscilalation_max_height])
    only_bg_y_data = np.array(y_data[y_counts_per_power < oscilalation_max_height])

    if exclude_points_around_peak is not None:
        index_maximum = np.argmax(y_data)
        only_bg_x_data = [only_bg_x_data[i] for i in range(len(only_bg_x_data))
                          if (i < index_maximum - exclude_points_around_peak)
                          or (i > index_maximum + exclude_points_around_peak)]
        only_bg_y_data = [only_bg_y_data[i] for i in range(len(only_bg_y_data))
                          if (i < index_maximum - exclude_points_around_peak)
                          or (i > index_maximum + exclude_points_around_peak)]

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
                 max_oscillation_amplitude_to_signal_ratio=0.3, exclude_points_around_peak=None, refractive_index=n_air,
                 want_oscillation=True,
                 key_to_be_power_corrected='spcm_mean_counts', key_of_power_correction='power_mean_uW',
                 force_power_ratio=True, power_background=0):
        """Chose either a folder or give a list of filenames"""
        self._set_filenames_and_manager(folder, filenames)

        self.varying_variable = varying_variable
        self.x_unit = self.varying_variable.split('(')[-1].split(')')[0]

        self.second_order = second_order
        self.refractive_index = refractive_index

        self.max_oscillation_amplitude_to_signal_ratio = max_oscillation_amplitude_to_signal_ratio
        self.exclude_points_around_peak = exclude_points_around_peak
        self.want_oscillation = want_oscillation

        self.key_to_be_power_corrected = key_to_be_power_corrected
        self.key_of_power_correction = key_of_power_correction
        key_numerator = '_'.join(self.key_to_be_power_corrected.split('_')[1:])
        key_denominator = self.key_of_power_correction.split('_')[-1]
        self.power_corrected_y_data_key = f'y_{key_numerator}_per_{key_denominator}'
        self.force_power_ratio = force_power_ratio
        self.power_background = power_background

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
            file_info = self.fnm.get_file_info_by_name(filename)
            for measurement_type in self.measurement_types_list:
                if file_info['Measurement Type'] == measurement_type:
                    self.measurement_filenames[measurement_type].append(filename)

    def _get_files(self):
        self.multi_objects = {}
        for measurement_type in self.measurement_filenames:
            filenames = self.measurement_filenames[measurement_type]
            if len(filenames):
                if measurement_type == 'power':
                    multi_object = self.measurement_types_classes[measurement_type](
                        filenames, force_power_ratio=self.force_power_ratio)
                else:
                    multi_object = self.measurement_types_classes[measurement_type](filenames)
                self.multi_objects[measurement_type]: Union[DataMultiPower, DataMultiSPCM] = multi_object

    def append_measurement(self, folder, filenames):
        if folder is not None:
            filenames = [path.join(folder, f) for f in os.listdir(folder)
                         if path.isfile(path.join(folder, f)) and f.endswith('.qdlf')]
            filenames.sort()

        if filenames != self.filenames:
            new_filenames = []
            for filename in filenames:
                if filename not in self.filenames:
                    self.filenames.append(filename)
                    new_filenames.append(filename)
            self.fnm = FilenameManager(self.filenames)  # TODO: Add an "append filename" function in FilenameManager
            self._get_measurement_filenames()

            for measurement_type in self.measurement_filenames:
                filenames = self.measurement_filenames[measurement_type]
                for new_filename in new_filenames:
                    if new_filename in filenames:
                        self.multi_objects[measurement_type].append_to_data_object_list(new_filename)

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
            file_info = self.fnm.get_file_info_by_name(filename)
            x_element = file_info[self.varying_variable]
            if file_info['Measurement Type'] == self.measurement_types_list[0]:  # to not double count
                x_data.append(x_element)

        if self.second_order is True:
            self.data['x_' + self.x_unit] = np.array(x_data) / 2
        else:
            self.data['x_' + self.x_unit] = np.array(x_data)

    def _set_other_x_data(self):
        if self.x_unit == 'nm':
            other_x_unit = 'eV'
        else:
            other_x_unit = 'nm'
        self.data['x_' + other_x_unit] = conversion_factor_nm_to_ev / (self.data['x_' + self.x_unit]
                                                                       * self.refractive_index)

    def _set_original_y_data(self):
        for measurement_type in self.multi_objects.keys():
            averages = self.multi_objects[measurement_type].averages
            stdevs = self.multi_objects[measurement_type].stdevs
            for key in averages:
                if key.startswith('y'):
                    key_init = key.split('_')[0]
                    unit = ''.join(key.split(key_init)[1:])
                    self.data[measurement_type + key_init[1:] + '_mean' + unit] = averages[key]
                    self.data[measurement_type + key_init[1:] + '_stdev' + unit] = stdevs[key]

    def _set_power_corrected_y_data(self):
        if self.key_of_power_correction in self.data.keys() and self.key_to_be_power_corrected in self.data.keys():
            self.data[self.power_corrected_y_data_key] = \
                self.data[self.key_to_be_power_corrected] / (
                            self.data[self.key_of_power_correction] - self.power_background)

    def _get_oscillation(self):
        if self.power_corrected_y_data_key in self.data.keys():
            x_data = np.array(self.data['x_nm'])
            if self.second_order:
                x_data = x_data * 2
            y_data = np.array(self.data[self.power_corrected_y_data_key])
            self.data['sine_correction'], self.sine_correction_phase = \
                get_oscillation(x_data, y_data, 'nm', self.max_oscillation_amplitude_to_signal_ratio,
                                self.exclude_points_around_peak)

    def _set_oscillation_corrected_y_data(self):
        if 'sine_correction' in self.data.keys():
            y_data = np.array(self.data[self.power_corrected_y_data_key])
            power_data = self.data[self.key_of_power_correction] - self.power_background
            sine_correction_data = np.array(self.data['sine_correction'])
            new_data_key = self.power_corrected_y_data_key.replace('y_', 'y_osc_corrected_')
            self.data[new_data_key] = y_data / sine_correction_data
            new_data_key = self.key_of_power_correction.replace('_', '_osc_corrected_', 1)
            self.data[new_data_key] = power_data * sine_correction_data


class DataAutoContinuousPLE:
    spacer = '_'
    measurement_types_list = ['power', 'spcm', 'wavelength']
    measurement_types_classes = {'power': DataPower, 'spcm': DataSPCM, 'wavelength': DataWavelength}
    allowed_file_extensions = ['mqdlf']
    default_keys = []
    allowed_units = unit_families
    qdlf_datatype = 'ContinuousWavelengthScanPLE'

    def __init__(self, filename, folder='./', second_order=True,
                 max_oscillation_amplitude_to_signal_ratio=0.3, exclude_points_around_peak=None, refractive_index=n_air,
                 want_oscillation=True,
                 type_wavelength_interpolation='simple', type_wavelength_interpolation_error='covariance'):
        """Give an mqdlf filename"""
        self.filename = path.join(folder, filename)

        self.second_order = second_order
        self.refractive_index = refractive_index

        self.max_oscillation_amplitude_to_signal_ratio = max_oscillation_amplitude_to_signal_ratio
        self.exclude_points_around_peak = exclude_points_around_peak
        self.want_oscillation = want_oscillation

        self.type_wavelength_interpolation = type_wavelength_interpolation
        self.type_wavelength_interpolation_error = type_wavelength_interpolation_error

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
        self._set_z_data()
        self._set_x_data()
        self._set_y_data()

    def _set_z_data(self):
        self.data['z_second'] = self.objects['spcm'].data['x_second']

    def _set_x_data(self):
        self._generate_wavelength_data()

    def _generate_wavelength_data(self):
        if self.type_wavelength_interpolation == 'simple':
            initial_wl = self.objects['wavelength'].data['y_nm'][0]
            final_wl = np.array(self.objects['wavelength'].data['y_nm'])[-2]
            max_object_data_length = max([len(self.objects[key].data['x_second']) for key in self.objects.keys()])

            # The following needs to be more intricate and include time
            self.data['x_nm'] = np.linspace(initial_wl, final_wl, max_object_data_length)

        elif self.type_wavelength_interpolation == 'linear_fit':
            self.interpolation_parameters, self.interpolation_fit_covariance = \
                curve_fit(line, self.objects['wavelength'].data['x_second'], self.objects['wavelength'].data['y_nm'])

            self.data['x_nm'] = line(self.objects['spcm'].data['x_second'], *self.interpolation_parameters)
        elif self.type_wavelength_interpolation == 'quadratic_fit':
            self.interpolation_parameters, self.interpolation_fit_covariance = \
                curve_fit(quadratic, self.objects['wavelength'].data['x_second'],
                          self.objects['wavelength'].data['y_nm'])

            self.data['x_nm'] = quadratic(self.objects['spcm'].data['x_second'], *self.interpolation_parameters)

        self.interpolation_error = {'nm': []}
        if self.type_wavelength_interpolation in ['linear_fit', 'quadratic_fit']:
            if self.type_wavelength_interpolation_error == 'covariance':
                self.interpolation_error['nm'] = [np.sqrt(self.interpolation_fit_covariance[i][i])
                                                  for i in range(len(self.interpolation_parameters))]

            elif self.type_wavelength_interpolation_error == 'maximum_likelihood':
                # from https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html
                # More info: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
                log_interpolation_error = 0
                if self.type_wavelength_interpolation == 'linear_fit':
                    def neg_log_likelihood(parameters, x_data, y_data):
                        a, b, log_sigma = parameters
                        sigma = np.exp(log_sigma)
                        delta = line(x_data, a, b) - y_data
                        return len(x_data) / 2 * np.log(2 * np.pi * sigma ** 2) + np.dot(delta, delta) / (
                                2 * sigma ** 2)

                    par_a, par_b, log_interpolation_error = minimize(neg_log_likelihood, (0, 0, 0), args=(
                        self.objects['wavelength'].data['x_second'], self.objects['wavelength'].data['y_nm'])).x

                elif self.type_wavelength_interpolation == 'quadratic_fit':
                    def neg_log_likelihood(parameters, x_data, y_data):
                        a, b, c, log_sigma = parameters
                        sigma = np.exp(log_sigma)
                        delta = quadratic(x_data, a, b, c) - y_data
                        return len(x_data) / 2 * np.log(2 * np.pi * sigma ** 2) + np.dot(delta, delta) / (
                                2 * sigma ** 2)

                    par_a, par_b, par_c, log_interpolation_error = minimize(neg_log_likelihood, (0, 0, 0, 0), args=(
                        self.objects['wavelength'].data['x_second'], self.objects['wavelength'].data['y_nm'])).x

                interpolation_error = np.exp(log_interpolation_error)
                self.interpolation_error['nm'] = interpolation_error
        else:
            self.interpolation_error['nm'] = None

        if self.second_order is True:
            self.data['x_nm'] /= 2
            try:
                self.interpolation_error['nm'] /= 2
            except TypeError:
                pass

        try:
            self.interpolation_error['eV'] = conversion_factor_nm_to_ev \
                                             / (np.mean(self.data['x_nm']) ** 2 * self.refractive_index) \
                                             * self.interpolation_error['nm']
        except TypeError:
            self.interpolation_error['eV'] = None

        self.data['x_eV'] = conversion_factor_nm_to_ev / (self.data['x_nm'] * self.refractive_index)

    def _set_y_data(self):
        # setting power data
        if 'power' in self.objects:
            self.power_univariate_spline = self.get_power_univariate_spline()
            self.power_cubic_spline = self.get_power_cubic_spline()
            self.data['y_power_unispline_uW'] = self.power_univariate_spline(self.data['z_second'])
            self.data['y_power_cubicspline_uW'] = self.power_cubic_spline(self.data['z_second'])
            if self.want_oscillation:
                self._get_oscillation()

        # setting spcm data
        if 'spcm' in self.objects:
            self.data['y_counts'] = self.objects['spcm'].data['y_counts']
            if 'power' in self.objects:
                self.data['y_counts_per_unispline_power'] = self.data['y_counts'] / self.data['y_power_unispline_uW']
                self.data['y_counts_per_cubicspline_power'] = self.data['y_counts'] / self.data[
                    'y_power_cubicspline_uW']
                if self.want_oscillation:
                    self._set_oscillation_corrected_y_data()

    def get_power_univariate_spline(self):
        power_df = self.objects['power'].data
        spline = UnivariateSpline(power_df['x_second'], power_df['y_uW'])

        return spline

    def get_power_cubic_spline(self):
        power_df = self.objects['power'].data
        spline = CubicSpline(power_df['x_second'], power_df['y_uW'])

        return spline

    def _get_oscillation(self):
        if 'y_power_cubicspline_uW' in self.data.keys():
            x_data = np.array(self.data['x_nm'])
            if self.second_order:
                x_data = x_data * 2
            y_data = np.array(self.data['y_power_cubicspline_uW'])
            self.data['sine_correction'], self.sine_correction_phase = \
                get_oscillation(x_data, y_data, 'nm', self.max_oscillation_amplitude_to_signal_ratio,
                                self.exclude_points_around_peak)

    def _set_oscillation_corrected_y_data(self):
        if 'sine_correction' in self.data.keys():
            if 'y_counts_per_unispline_power' in self.data.keys():
                y_data = np.array(self.data['y_counts_per_unispline_power'])
                power_data = self.data['y_power_unispline_uW']
                sine_correction_data = np.array(self.data['sine_correction'])
                self.data['y_osc_corrected_counts_per_unispline_power'] = y_data / sine_correction_data
                self.data['y_osc_corrected_power_unispline_uW'] = power_data * sine_correction_data
            if 'y_counts_per_cubicspline_power' in self.data.keys():
                y_data = np.array(self.data['y_counts_per_cubicspline_power'])
                power_data = self.data['y_power_cubicspline_uW']
                sine_correction_data = np.array(self.data['sine_correction'])
                self.data['y_osc_corrected_counts_per_cubicspline_power'] = y_data / sine_correction_data
                self.data['y_osc_corrected_power_cubicspline_uW'] = power_data * sine_correction_data
