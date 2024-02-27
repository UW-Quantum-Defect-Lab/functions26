# 2021-01-27
# This code was made for use in the Fu lab
# by Christian Zimmermann

# commented and updated on 05-10-2022

# Import libraries
import warnings
from typing import Dict, Union

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio
import scipy.optimize as spo
import scipy.signal as sps
import scipy.special as spsp
import scipy.interpolate as spi

from matplotlib.colors import LogNorm  # , Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .constants import conversion_factor_nm_to_ev  # eV*nm
from .constants import n_air
from .DataDictXXX import DataDictFilenameInfo

from .FittingManager import FittingManager, voigt_linear_fit

# Define some helper functions
# This function converts strings used to denote positions on confocal images into floats
# Example: This converts 'n3p22' to -3.22 or '2p20' to 2.2
def convert_to_string_to_float(number_string):
    # Figure out sign, 'n' or 'm' denote minus
    if 'n' in number_string:
        sign = -1
        number_string = number_string.split('n')[1]
    elif 'm' in number_string:
        sign = -1
        number_string = number_string.split('m')[1]
    else:
        sign = 1

    # Figure out decimal point, 'p' denotes decimal point
    if 'p' in number_string:
        number = float(number_string.replace('p', '.'))
    else:
        number = float(number_string)

    # Apply sign
    number *= sign

    return number


# This function converts list of numbers (x and y coordinate) into a position string used in this class
# convention for position strings, example: 'xn3p22_y2p20' is x = -3.22 and y = 2.2
def convert_xy_to_position_string(position):
    # We are using x and y axes
    axes = ['x', 'y']

    # Construct string, use 'n' for negative numbers and 'p' for decimal points
    string = ''
    for n, axis in enumerate(axes):
        string += axis
        if position[n] < 0:
            position[n] = np.abs(position[n])
            sign_string = 'n'
        else:
            sign_string = ''
        string += sign_string
        string += str(np.round(position[n], 3)).replace('.', 'p')
        if n == 0:
            string += '_'

    return string


# Define line with slope a and intersect b
def line(x, a, b):
    return a * x + b


# Define an exponential with some parameters
def exponential(x, a, b, x0):
    return a * np.exp(b * (x - x0))


# Define function to guess initial parameters for fitting certain functions
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
    else:
        warnings.warn(f'function {function} is not an option. Try one of: "linear", "exponential".')
        p0 = None

    return p0


# Define function for a linear baseline and 2 voigt functions
# This function is used when analyzing spectra that contain two or one sharp peak
def two_voigt_and_linear_baseline(x,
                                  slope, intersect,
                                  amplitude_1, width_gaussian_1, width_lorentzian_1, position_1,
                                  amplitude_2, width_gaussian_2, width_lorentzian_2, position_2):
    return (line(x, slope, intersect)
            + amplitude_1 * spsp.voigt_profile(x - position_1, width_gaussian_1, width_lorentzian_1)
            + amplitude_2 * spsp.voigt_profile(x - position_2, width_gaussian_2, width_lorentzian_2))


# This function can be used to guess starting parameters for a linear fit using two spectral
# ranges at the edge of another spectral range:
# [Fitting Range for Line][Spectral Range of Interest][Fitting Range for Line]
def guess_linear_fit(sub_spectrum,
                     unit_spectral_range='eV',
                     number_of_points=10):
    sub_spectrum.reset_index(drop=True, inplace=True)
    sub_spectrum_left = sub_spectrum.loc[0:number_of_points]
    sub_spectrum_right = sub_spectrum.loc[len(sub_spectrum.index) - number_of_points + 1:len(sub_spectrum.index) + 1]
    sub_spectrum_left.reset_index(drop=True, inplace=True)
    sub_spectrum_right.reset_index(drop=True, inplace=True)
    slope = (sub_spectrum_right['y_counts_per_seconds'][0] - sub_spectrum_left['y_counts_per_seconds'][0]) / (
            sub_spectrum_right['x_{0}'.format(unit_spectral_range)][0] -
            sub_spectrum_left['x_{0}'.format(unit_spectral_range)][0])
    intersect = sub_spectrum_left['y_counts_per_seconds'][0] - slope * \
                sub_spectrum_left['x_{0}'.format(unit_spectral_range)][0]
    sub_spectrum_edges = pd.concat([sub_spectrum_left, sub_spectrum_right], ignore_index=True)
    params, covar = spo.curve_fit(line, sub_spectrum_edges['y_counts_per_seconds'],
                                  sub_spectrum_edges['x_{0}'.format(unit_spectral_range)], p0=[slope, intersect])
    slope = params[0]
    intersect = params[1]
    return slope, intersect


# This function finds the maximum position of spectrum[y_axis] in spectrum['x_axis'] using interpolation
def find_maximum_position_interpolated(spectrum,
                                       x_axis='eV',
                                       y_axis='counts_per_seconds',
                                       interpolation_type='linear',
                                       # Can be any method scipi.interpolate.interp1d allows
                                       nan_value=np.NaN,  # If the maximum position can't be found return nan_value
                                       intensity_limit=5000
                                       ):
    x_data = spectrum['x_{0}'.format(x_axis)].to_list()
    y_data = spectrum['y_{0}'.format(y_axis)].to_list()

    interpolation_func = spi.interp1d(x_data, y_data,
                                      kind=interpolation_type)

    index_maximum = np.argmax(y_data)

    try:
        x_interpolation = np.linspace(x_data[index_maximum - 2], x_data[index_maximum + 2], 1000)

        y_interpolation = interpolation_func(x_interpolation)

        if np.max(y_interpolation) < intensity_limit:
            return nan_value

        index_maximum_interpolation = np.argmax(y_interpolation)

        return x_interpolation[index_maximum_interpolation]
    except IndexError:
        return nan_value


# This function finds the full width at the n-th fraction of the maximum
# Example: n = 2 yields the full width at half maximum
# Interpolation is used for this too
def find_full_width_atnfraction_maximum_interpolated(spectrum,
                                                     x_axis='eV',
                                                     fraction=2,  # That's full width at half maximum
                                                     y_axis='counts_per_seconds',
                                                     interpolation_type='linear',
                                                     # Can be any method scipi.interpolate.interp1d allows
                                                     nan_value=np.NaN
                                                     # If the maximum position can't be found return nan_value
                                                     ):
    x_data = np.array(spectrum['x_{0}'.format(x_axis)].to_list())
    y_data = np.array(spectrum['y_{0}'.format(y_axis)].to_list())

    interpolation_func = spi.interp1d(x_data, y_data,
                                      kind=interpolation_type)

    index_maximum = np.argmax(y_data)
    try:
        # Find boundaries of width to be found from data
        x_data_right = x_data[: index_maximum + 1]
        y_data_right = y_data[: index_maximum + 1]
        x_data_left = x_data[index_maximum:]
        y_data_left = y_data[index_maximum:]

        diff_y_left = np.abs(y_data_left - y_data[index_maximum] / fraction)
        diff_y_right = np.abs(y_data_right - y_data[index_maximum] / fraction)

        index_left = np.argmin(diff_y_left)
        index_right = np.argmin(diff_y_right)

        # Find better width using interpolation
        x_data_left_interpolation = np.linspace(x_data_left[index_left - 2], x_data[index_left + 2], 1000)
        x_data_right_interpolation = np.linspace(x_data_right[index_right - 2], x_data[index_right + 2], 1000)

        y_data_left_interpolation = interpolation_func(x_data_left)
        y_data_right_interpolation = interpolation_func(x_data_right)

        diff_y_left_interpolation = np.abs(y_data_left_interpolation - y_data[index_maximum] / fraction)
        diff_y_right_interpolation = np.abs(y_data_right_interpolation - y_data[index_maximum] / fraction)

        index_left_interpolation = np.argmin(diff_y_left_interpolation)
        index_right_interpolation = np.argmin(diff_y_right_interpolation)

        width = np.abs(
            x_data_left_interpolation[index_left_interpolation] - x_data_right_interpolation[index_right_interpolation])

        return width
    except IndexError:
        return nan_value


# Define base class for importing confocal image data
class DataImage:

    def __init__(self, file_name, folder_name, allowed_file_extensions):
        self.file_name = file_name
        if file_name == '':
            raise RuntimeError('File name is an empty string')
        self.folder_name = folder_name
        self.file_extension = self.file_name.split('.')[-1]
        self.check_file_type(allowed_file_extensions)

        self.file_info = DataDictFilenameInfo()
        self.get_file_info()

        self.image_data = np.zeros((1, 1))
        self.extent = {'x_min': 0., 'x_max': 0., 'y_min': 0., 'y_max': 0.}
        self.get_data()

    def get_data(self):
        warnings.warn('Define your own get_data() function')
        pass

    def get_file_info(self):

        # Save filename without folder and file extension
        file_info_raw = self.file_name.split('.')[-2]
        if '/' in self.file_name:
            file_info_raw = file_info_raw.split('/')[-1]

        file_info_raw_components = file_info_raw.split('_')  # All file info are separated by '_'
        self.file_info.get_info(file_info_raw_components)  # retrieve info from file

        return True

    def check_file_type(self, allowed_file_extensions):
        allowed_file_extensions = [fe.lower() for fe in allowed_file_extensions]
        if self.file_extension.lower() not in allowed_file_extensions:
            raise RuntimeError('Given file extension does not much the allowed extensions: '
                               + str(allowed_file_extensions))


# Define class for importing confocal image data (including spectral confocal scans)
class DataConfocalScan(DataImage):
    allowed_file_extensions = ['mat']

    def __init__(self,
                 file_name,
                 folder_name='.',
                 spectral_range='all',
                 # no specific spectral range is defined for analysis (only for spectral confocal scan data)
                 unit_spectral_range='eV',
                 baseline=None,
                 # if baseline is not None, a baseline will be substracted from
                 # spectral data (options: 'linear', 'exponential')
                 nan_value=np.NaN,  # Default pixel value if analysis fails for whatever reason
                 method='sum',  # See line 477 ff. for options
                 intensity_threshold=None,
                 # If max value of data.y is lower than intensity_threshold, pixel is set to nan_value
                 background=300,  # electronic background
                 wavelength_offset=0,  # wavelength offset in nm (raw data!)
                 new_wavelength_axis=None,
                 # Import x axis if the correct x axis wasn't imported before running the scan
                 second_order=True,  # Was the spectrometer placed at the 2nd or 1st order of diffraction?
                 refractive_index=n_air  # Use get_nair from functions26 for n_air at a specific wavelength
                 ):

        # Add parameters to self
        self.spectral_range = spectral_range
        self.unit_spectral_range = unit_spectral_range
        self.background = background
        self.second_order = second_order
        self.refractive_index = refractive_index
        self.wavelength_offset = wavelength_offset
        self.new_wavelength_axis = new_wavelength_axis
        self.method = method
        self.intensity_threshold = intensity_threshold
        self.nan_value = nan_value

        # Construct baseline info, if baseline was chosen
        # baseline string: example: linear or linear_edge-5_edge-5
        self.baseline = {}
        if baseline is not None:
            if '_' in baseline:
                baseline_keyword_components = baseline.split('_')
                baseline_type = baseline_keyword_components[0]
                baseline_method_left = baseline_keyword_components[1].split('-')[0]
                baseline_points_left = int(baseline_keyword_components[1].split('-')[1])
                baseline_method_right = baseline_keyword_components[2].split('-')[0]
                baseline_points_right = int(baseline_keyword_components[2].split('-')[1])
            else:
                baseline_type = baseline
                baseline_method_left = 'edge'
                baseline_points_left = 20
                baseline_method_right = 'edge'
                baseline_points_right = 20
            self.baseline['type'] = baseline_type
            self.baseline['method_left'] = baseline_method_left
            self.baseline['method_right'] = baseline_method_right
            self.baseline['points_left'] = baseline_points_left
            self.baseline['points_right'] = baseline_points_right
        else:
            self.baseline['type'] = None

        super().__init__(file_name, folder_name, self.allowed_file_extensions)

    # Get data from file
    # This includes analysis of spectral confocal scan according to the parameters that were set
    def get_data(self):

        # Get raw data from matlab file
        matlab_file_data = spio.loadmat(self.file_name)

        if 'scan' in matlab_file_data.keys():
            self.software = 'DoritoScopeConfocal'
            self.image_data = matlab_file_data['scan'][0][0][4]
            self.exposure_time = matlab_file_data['scan'][0][0][11][0][0]
            self.image_data = self.image_data / self.exposure_time
            # Convert image, so it looks like what we see in the matlab GUI
            self.image_data = np.transpose(self.image_data)
            self.image_data = np.flip(self.image_data, axis=0)

            self.x = matlab_file_data['scan'][0][0][0][0]
            self.y = matlab_file_data['scan'][0][0][1][0]
            self.y = np.flip(self.y)
        elif 'data' in matlab_file_data.keys():
            self.software = 'McDiamond'
            self.image_data = matlab_file_data['data'][0][0][7][0][0]
            # Convert image, so it looks like what we see in the matlab GUI
            self.image_data = np.transpose(self.image_data)
            self.image_data = np.flip(self.image_data, axis=0)

            self.x = matlab_file_data['data'][0][0][2][0][0][0]
            self.y = matlab_file_data['data'][0][0][2][0][1][0]

        self.extent['x_min'] = np.min(self.x)
        self.extent['x_max'] = np.max(self.x)
        self.extent['y_min'] = np.min(self.y)
        self.extent['y_max'] = np.max(self.y)

        # Shift all values by half a pixel to have x, y position be associated with pixel center
        x_pixel_length = (self.extent['x_max'] - self.extent['x_min']) / (len(self.x) - 1)
        self.extent['x_min'] = self.extent['x_min'] - x_pixel_length / 2
        self.extent['x_max'] = self.extent['x_max'] + x_pixel_length / 2
        y_pixel_length = (self.extent['y_max'] - self.extent['y_min']) / (len(self.y) - 1)
        self.extent['y_min'] = self.extent['y_min'] - y_pixel_length / 2
        self.extent['y_max'] = self.extent['y_max'] + y_pixel_length / 2

        # Check whether spectrometer was used for data collection, if yes also import spectra
        if self.software == 'DoritoScopeConfocal':
            if matlab_file_data['scan'][0][0][3][0] == 'Spectrometer':
                self.type = 'Spectrometer'
                self.exposure_time = matlab_file_data['scan'][0][0][11][0][0]
                self.cycles = matlab_file_data['scan'][0][0][10][0][0]
                spectra_raw = matlab_file_data['scan'][0][0][15]
                self.spectra_raw = [[spectra_raw[ix][iy][0] for iy in range(len(self.y))] for ix in range(len(self.x))]
                self.spectra_raw = np.transpose(self.spectra_raw, axes=[1, 0, 2])
                self.spectra_raw = np.flip(self.spectra_raw, axis=0)
                self.spectra_raw = self.spectra_raw - self.background
                self.spectra_raw = self.spectra_raw / self.exposure_time
                if self.new_wavelength_axis is not None:
                    self.wavelength = self.new_wavelength_axis + self.wavelength_offset
                else:
                    self.wavelength = matlab_file_data['scan'][0][0][16][0] + self.wavelength_offset
                if self.second_order:
                    self.wavelength = self.wavelength / 2
                self.photon_energy = conversion_factor_nm_to_ev / (self.wavelength * self.refractive_index)
                self.spectra = {}

                # Spectra are saved in dict with positions strings as keys
                for ix, x_position in enumerate(self.x):
                    for iy, y_position in enumerate(self.y):
                        position_string = convert_xy_to_position_string([x_position, y_position])
                        self.spectra[position_string] = pd.DataFrame(
                            data={'x_nm': self.wavelength, 'y_counts_per_seconds': self.spectra_raw[iy][ix]})
                        self.spectra[position_string]['x_eV'] = self.photon_energy

                # Construct image from spectra
                self.image_data_from_spectra = []
                self.sub_spectra = {}
                for ix, x_position in enumerate(self.x):
                    counts_for_image_along_y = []
                    for iy, y_position in enumerate(self.y):
                        position_string = convert_xy_to_position_string([x_position, y_position])
                        spectrum = pd.DataFrame(
                            data={'x_nm': self.wavelength, 'y_counts_per_seconds': self.spectra_raw[iy][ix]})
                        spectrum['x_eV'] = self.photon_energy

                        if self.spectral_range != 'all':
                            # Get sub spectra needed for baseline
                            if self.baseline['type'] is not None:
                                self.baseline[position_string] = {}
                                if self.baseline['method_left'] == 'edge':
                                    index_left = np.abs(
                                        spectrum['x_{0}'.format(self.unit_spectral_range)] - self.spectral_range[
                                            0]).idxmin()
                                elif self.baseline['method_left'] == 'minimum':
                                    index_left = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >=
                                                               self.spectral_range[0]) & (
                                                                      spectrum[
                                                                          'x_{0}'.format(self.unit_spectral_range)] <=
                                                                      self.spectral_range[1])][
                                        'y_counts_per_seconds'].idxmin()
                                elif self.baseline['method_left'] == 'maximum':
                                    index_left = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >=
                                                               self.spectral_range[0]) & (
                                                                      spectrum[
                                                                          'x_{0}'.format(self.unit_spectral_range)] <=
                                                                      self.spectral_range[1])][
                                        'y_counts_per_seconds'].idxmax()
                                if self.baseline['method_right'] == 'edge':
                                    index_right = np.abs(
                                        spectrum['x_{0}'.format(self.unit_spectral_range)] - self.spectral_range[
                                            1]).idxmin()
                                elif self.baseline['method_right'] == 'minimum':
                                    index_right = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >=
                                                                self.spectral_range[0]) & (
                                                                       spectrum[
                                                                           'x_{0}'.format(self.unit_spectral_range)] <=
                                                                       self.spectral_range[1])][
                                        'y_counts_per_seconds'].idxmin()
                                elif self.baseline['method_right'] == 'maximum':
                                    index_right = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >=
                                                                self.spectral_range[0]) & (
                                                                       spectrum[
                                                                           'x_{0}'.format(self.unit_spectral_range)] <=
                                                                       self.spectral_range[1])][
                                        'y_counts_per_seconds'].idxmax()
                                sub_spectrum_for_fitting_left = spectrum.loc[
                                                                index_left - self.baseline['points_left']:
                                                                index_left + self.baseline['points_left']]
                                sub_spectrum_for_fitting_right = spectrum.loc[
                                                                 index_right - self.baseline['points_right']:
                                                                 index_right + self.baseline['points_right']]

                                # Save sub spectra for baseline
                                self.baseline[position_string]['sub_spectrum_for_fitting'] = pd.concat(
                                    [sub_spectrum_for_fitting_left, sub_spectrum_for_fitting_right], ignore_index=True)
                                self.baseline[position_string][
                                    'sub_spectrum_for_fitting_left'] = sub_spectrum_for_fitting_left
                                self.baseline[position_string][
                                    'sub_spectrum_for_fitting_right'] = sub_spectrum_for_fitting_right

                                # Get baselines, if baseline fit fails (RunTimeError), set pixel to nan_value
                                try:
                                    set_pixel_to_zero = False
                                    if self.baseline['type'] == 'linear':
                                        p0 = guess_initial_parameters(
                                            self.baseline[position_string]['sub_spectrum_for_fitting'][
                                                'x_{0}'.format(self.unit_spectral_range)],
                                            self.baseline[position_string]['sub_spectrum_for_fitting'][
                                                'y_counts_per_seconds'], 'linear')
                                        parameters, covariance = spo.curve_fit(line, self.baseline[position_string][
                                            'sub_spectrum_for_fitting']['x_{0}'.format(self.unit_spectral_range)],
                                                                               self.baseline[position_string][
                                                                                   'sub_spectrum_for_fitting'][
                                                                                   'y_counts_per_seconds'], p0=p0)
                                        self.baseline[position_string]['slope_initial'] = p0[0]
                                        self.baseline[position_string]['intersect_initial'] = p0[1]
                                        self.baseline[position_string]['slope'] = parameters[0]
                                        self.baseline[position_string]['intersect'] = parameters[1]
                                    elif self.baseline['type'] == 'minimum':
                                        self.baseline[position_string]['offset'] = spectrum[
                                            'y_counts_per_seconds'].min()
                                    elif self.baseline['type'] == 'exponential':
                                        p0 = guess_initial_parameters(
                                            self.baseline[position_string]['sub_spectrum_for_fitting'][
                                                'x_{0}'.format(self.unit_spectral_range)],
                                            self.baseline[position_string]['sub_spectrum_for_fitting'][
                                                'y_counts_per_seconds'], 'exponential')
                                        parameters, covariance = spo.curve_fit(exponential,
                                                                               self.baseline[position_string][
                                                                                   'sub_spectrum_for_fitting'][
                                                                                   'x_{0}'.format(
                                                                                       self.unit_spectral_range)],
                                                                               self.baseline[position_string][
                                                                                   'sub_spectrum_for_fitting'][
                                                                                   'y_counts_per_seconds'], p0=p0)
                                        self.baseline[position_string]['prefactor_initial'] = p0[0]
                                        self.baseline[position_string]['exponent_initial'] = p0[1]
                                        self.baseline[position_string]['shift_initial'] = p0[2]
                                        self.baseline[position_string]['prefactor'] = parameters[0]
                                        self.baseline[position_string]['exponent'] = parameters[1]
                                        self.baseline[position_string]['shift'] = parameters[2]
                                except RuntimeError:
                                    set_pixel_to_zero = True

                            # Get spectrum in bounds defined by self.spectral_range
                            spectrum = spectrum.loc[
                                (spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                        spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]
                            self.sub_spectra[position_string] = spectrum

                            # One method requires taking another spectral range besides self.spectral_range into account
                            # Fetch secondary spectrum
                            if 'divide_by_sum_in' in self.method:
                                bounds = [float(self.method.split('_')[-2]), float(self.method.split('_')[-1])]
                                secondary_spectrum = spectrum.loc[
                                    (spectrum['x_{0}'.format(self.unit_spectral_range)] >= bounds[0]) & (
                                            spectrum['x_{0}'.format(self.unit_spectral_range)] <= bounds[1])]

                            # Subtract baseline from spectral data
                            if self.baseline['type'] is not None and not set_pixel_to_zero:
                                self.baseline[position_string]['x'] = spectrum['x_{0}'.format(self.unit_spectral_range)]
                                if self.baseline['type'] == 'linear':
                                    self.baseline[position_string]['y'] = line(self.baseline[position_string]['x'],
                                                                               self.baseline[position_string]['slope'],
                                                                               self.baseline[position_string][
                                                                                   'intersect'])
                                    self.baseline[position_string]['y_initial'] = line(
                                        self.baseline[position_string]['x'],
                                        self.baseline[position_string]['slope_initial'],
                                        self.baseline[position_string]['intersect_initial'])
                                elif self.baseline['type'] == 'exponential':
                                    self.baseline[position_string]['y'] = exponential(
                                        self.baseline[position_string]['x'],
                                        self.baseline[position_string]['prefactor'],
                                        self.baseline[position_string]['exponent'],
                                        self.baseline[position_string]['shift'])
                                    self.baseline[position_string]['y_initial'] = exponential(
                                        self.baseline[position_string]['x'],
                                        self.baseline[position_string]['prefactor_initial'],
                                        self.baseline[position_string]['exponent_initial'],
                                        self.baseline[position_string]['shift_initial'])
                                elif self.baseline['type'] == 'minimum':
                                    self.baseline[position_string]['y'] = line(self.baseline[position_string]['x'], 0,
                                                                               self.baseline[position_string]['offset'])
                                spectrum['y_counts_per_seconds'] = spectrum['y_counts_per_seconds'] - \
                                                                   self.baseline[position_string]['y']

                        # Fetch/Calculate pixel values (This will be done for all pixels if self.intensity_threshold is
                        # None or pixels with max counts above self.intensity_threshold)
                        if self.intensity_threshold is None:
                            find_pixel_value = True
                        else:
                            if np.max(spectrum['y_counts_per_seconds']) >= self.intensity_threshold:
                                find_pixel_value = True
                            else:
                                find_pixel_value = False
                        # Fetch/Calculate pixel values
                        if find_pixel_value:
                            # Sum counts in chosen spectral range
                            if self.method == 'sum':
                                counts_for_image = spectrum['y_counts_per_seconds'].sum()
                            # Find maximum intensity in chosen spectral range
                            elif self.method == 'maximum_intensity':
                                counts_for_image = np.max(spectrum['y_counts_per_seconds'])
                            # Find maximum position in chosen spectral range
                            elif self.method == 'maximum_position':
                                index = spectrum['y_counts_per_seconds'].idxmax()
                                counts_for_image = spectrum.loc[index, 'x_{0}'.format(self.unit_spectral_range)]
                            elif self.method == 'sum_with_baseline':
                                x_data = np.array(spectrum['x_eV'])
                                y_data = np.array(spectrum['y_counts_per_seconds'])
                                N_points = len(x_data)
                                if N_points < 3:
                                    raise ValueError('The number of points in this spectral region is less than 3.')

                                # x_data, y_data = (x_data[2:-2], np.convolve(y_data, np.ones(5), 'valid') / 5)
                                sss = y_data.sum()
                                N_split = N_points//3
                                Dx = np.diff(x_data)
                                Dy = np.diff(y_data)
                                slope = np.average(Dy)/np.average(Dx)
                                const = 0.5 * np.average(y_data[:N_split] - slope * x_data[:N_split]) + 0.5 * np.average(y_data[-N_split:] - slope * x_data[-N_split:])
                                y_data = y_data - slope * x_data - const

                                

                                x_roll, y_roll = (x_data[1:-1], np.convolve(y_data, np.ones(3), 'valid') / 3)
                                # counts_for_image = y_data[y_roll[N_split:-N_split].argmax() + 1]
                                # counts_for_image = np.max(y_roll) * 5
                                counts_for_image = y_data.max() * 3
                                # counts_for_image = sss - y_data.sum()

                                # try:
                                #     model_names = ['quadratic', 'voigt']
                                #     model_prefixes = ['', '']
                                #     models_df = pd.DataFrame({'names': model_names, 'prefixes': model_prefixes})
                                #     input_parameters=pd.DataFrame({'names': ['center', 'amplitude'], 
                                #                                    'initial_values': [x_data[np.argmax(y_data)], np.max(y_data)], 
                                #                                    'is_fixed': [False, False], 
                                #                                    'bounds': [[x_data[len(y_data)//3] - x_data.mean(), x_data[-len(y_data)//3] - x_data.mean()], [0, 3 * np.max(y_data)]]})
                                #     fitmng = FittingManager(x_data - x_data.mean(), y_data, models_df, input_parameters=input_parameters)
                                #     # x_fit, y_fit = fitmng.get_x_y_fit(x_data.min(), x_data.max(), len(x_data))
                                #     counts_for_image = fitmng.fit_pars['amplitude']
                                # #     counts_for_image = np.sum(y_data) - np.sum(y_fit)
                                # # #     fitmng, fwhm, center = voigt_linear_fit(x_data, y_data)
                                # # #     counts_for_image = fitmng.fit_pars['amplitude']
                                # except Exception as e:
                                #     print(e)
                                #     counts_for_image = self.nan_value

                            # Sum intensity of a triplet
                            # The triplet are the peak within a chosen spectral range and the
                            # intensity of its two neighbours
                            # This method is particularly good in picking up sharp single emitters in spectral data
                            elif self.method == 'maximum_intensity_triplet':
                                index = spectrum['y_counts_per_seconds'].idxmax()
                                try:
                                    counts_for_image = (spectrum.loc[index, 'y_counts_per_seconds'] +
                                                        spectrum.loc[index - 1, 'y_counts_per_seconds'] +
                                                        spectrum.loc[index + 1, 'y_counts_per_seconds'])
                                # Error occurs of maximum is right at one of the edges of the spectral range
                                except (IndexError, KeyError) as e:
                                    counts_for_image = self.nan_value

                            # Sum intensity of a triplet
                            # The triplet are the peak within a chosen spectral range and the intensity of its
                            # two neighbours
                            # This method is particularly good at picking up shar single emitters in spectral data
                            # Here, a baseline will be subtracted, the baseline will be calculated from a range
                            # relative to the peak
                            elif 'maximum_intensity_triplet_with_baseline' in self.method:
                                index = spectrum['y_counts_per_seconds'].idxmax()

                                # baseline is fitted to a range found based on the peak position
                                # baseline_start states how far away from the peak the baseline fit starts
                                baseline_start = int(self.method.split('maximum_intensity_triplet_with_baseline_')[-1])
                                indexes = [index + i for i in [baseline_start + j for j in range(4)]] + \
                                          [index - i for i in [baseline_start + j for j in range(4)]]
                                try:
                                    spectrum_for_baseline = spectrum.loc[indexes]
                                    p0 = guess_initial_parameters(
                                        spectrum_for_baseline['x_{0}'.format(self.unit_spectral_range)],
                                        spectrum_for_baseline['y_counts_per_seconds'], 'linear')
                                    parameters, covariance = spo.curve_fit(line, spectrum_for_baseline[
                                        'x_{0}'.format(self.unit_spectral_range)],
                                                                           spectrum_for_baseline[
                                                                               'y_counts_per_seconds'], p0=p0)
                                    baseline = line(spectrum['x_{0}'.format(self.unit_spectral_range)], *parameters)
                                    spectrum['y_counts_per_seconds'] = spectrum['y_counts_per_seconds'] - baseline
                                    counts_for_image = (spectrum.loc[index, 'y_counts_per_seconds'] +
                                                        spectrum.loc[index - 1, 'y_counts_per_seconds'] +
                                                        spectrum.loc[index + 1, 'y_counts_per_seconds'])
                                # Error occurs of maximum is right at one of the edges of the spectral range
                                except (IndexError, KeyError) as e:
                                    counts_for_image = self.nan_value

                            # See method above
                            # Here, the maximum position itself is found after subtracting a baseline
                            elif 'maximum_position_single_with_baseline' in self.method:
                                index = spectrum['y_counts_per_seconds'].idxmax()

                                components = self.method.split('maximum_position_single_with_baseline_')[-1]
                                baseline_start = int(components.split('_')[-2])
                                print(baseline_start)
                                indexes = [index + i for i in [baseline_start + j for j in range(4)]] + \
                                          [index - i for i in [baseline_start + j for j in range(4)]]

                                lower_limit = float(components.split('_')[-1])
                                try:
                                    spectrum_for_baseline = spectrum.loc[indexes]
                                    p0 = guess_initial_parameters(
                                        spectrum_for_baseline['x_{0}'.format(self.unit_spectral_range)],
                                        spectrum_for_baseline['y_counts_per_seconds'], 'linear')
                                    parameters, covariance = spo.curve_fit(line, spectrum_for_baseline[
                                        'x_{0}'.format(self.unit_spectral_range)],
                                                                           spectrum_for_baseline[
                                                                               'y_counts_per_seconds'], p0=p0)
                                    baseline = line(spectrum['x_{0}'.format(self.unit_spectral_range)], *parameters)
                                    spectrum['y_counts_per_seconds'] = spectrum['y_counts_per_seconds'] - baseline
                                    if (spectrum.loc[index, 'y_counts_per_seconds'] + spectrum.loc[
                                        index - 1, 'y_counts_per_seconds'] + spectrum.loc[
                                            index + 1, 'y_counts_per_seconds']) > lower_limit:
                                        counts_for_image = spectrum.loc[index, 'x_{0}'.format(self.unit_spectral_range)]
                                    else:
                                        counts_for_image = self.nan_value
                                except (IndexError, KeyError) as e:
                                    counts_for_image = self.nan_value

                            # The maximum position within a chosen spectral range is found using interpolation to get
                            # a better value for the position
                            elif self.method == 'maximum_position_interpolated':
                                counts_for_image = find_maximum_position_interpolated(
                                    spectrum, x_axis=self.unit_spectral_range, nan_value=self.nan_value)

                            # The full width at half maximum of a feature within a chosen range is found using
                            # interpolation to get a better value
                            elif self.method == 'full_width_at_half_maximum_interpolated':
                                counts_for_image = find_full_width_atnfraction_maximum_interpolated(
                                    spectrum, x_axis=self.unit_spectral_range, nan_value=self.nan_value)

                            # The center of mass position within a chosen spectral range is calculated
                            # All values below threshold will be set to 0 for the weighted average (suppresses noise)
                            elif 'center_of_mass_position' in self.method:
                                threshhold = float(self.method.split('center_of_mass_position_')[-1])
                                weights = []
                                for y in spectrum['y_counts_per_seconds']:
                                    if y > threshhold:
                                        weights.append(y)
                                    else:
                                        weights.append(0)
                                counts_for_image = np.average(
                                    spectrum['x_{0}'.format(self.unit_spectral_range)].to_numpy(), weights=weights)

                            # The sum in self.spectral range will be divided by the sum within a second spectral range
                            elif 'divide_by_sum_in' in self.method:
                                counts_for_image = spectrum['y_counts_per_seconds'].sum() / \
                                                   secondary_spectrum['y_counts_per_seconds'].sum()

                            # Tnis method was developed for analyzing the D0X lines in the fibbed ZnO slice (spectral
                            # position of the lines changes from pixel to pixel)
                            elif 'ShiftByMaxIn' in self.method and 'ShiftTo' in self.method \
                                    and 'IntegrateIn' in self.method:
                                # Get range used to calculate shift (shift by maximum position)
                                # concention: ShiftByMaxIn_X-Y
                                shift_range_str = self.method.split('ShiftByMaxIn_')[-1].split('_')[0]
                                shift_range = [float(shift_range_str.split('-')[0]),
                                               float(shift_range_str.split('-')[1])]

                                # Get value spectra should be shifted to
                                E0 = float(self.method.split('ShiftTo_')[-1].split('_')[0])

                                # Get range(s) to sum up (this refers to spectral range after the shift!)
                                # ~ will separate several ranges
                                # convention: IntegrateIn_X1-Y1~X2-Y2 etc.
                                integration_ranges_str = self.method.split('IntegrateIn_')[-1].split('_')[0]
                                _ = integration_ranges_str.split('~')
                                integration_ranges = []
                                for s in _:
                                    integration_ranges.append([float(s.split('-')[0]), float(s.split('-')[1])])

                                if 'DivideByIn_' in self.method:
                                    # If 'DivideByIn' is in self.method, also fetch those ranges
                                    # ~ will separate several ranges
                                    # convention: DivideByIn_X1-Y1~X2-Y2 etc.
                                    division_ranges_str = self.method.split('DivideByIn_')[-1].split('_')[0]
                                    _ = division_ranges_str.split('~')
                                    division_ranges = []
                                    for s in _:
                                        division_ranges.append([float(s.split('-')[0]), float(s.split('-')[1])])

                                # Find shift
                                sub_data = spectrum.loc[
                                    (spectrum['x_eV'] >= shift_range[0]) & (spectrum['x_eV'] <= shift_range[1])]
                                sub_data.reset_index(inplace=True, drop=True)

                                index_shift = sub_data['y_counts_per_seconds'].argmax()
                                shift = sub_data.loc[index_shift, 'x_eV']

                                # Shift data (on a copy of spectrum, for not over-writing spectra)
                                data = spectrum.copy()
                                data['x_eV'] = data['x_eV'] - shift + E0

                                # Integrate data for numerator
                                integrated_counts_numerator = 0
                                for sr in integration_ranges:
                                    sub_data = data.loc[(data['x_eV'] > sr[0]) & (data['x_eV'] < sr[1])]
                                    integrated_counts_numerator += np.sum(sub_data['y_counts_per_seconds'])

                                # Integrate data for denominator
                                if 'DivideByIn_' in self.method:
                                    integrated_counts_denominator = 0
                                    for sr in division_ranges:
                                        sub_data = data.loc[(data['x_eV'] > sr[0]) & (data['x_eV'] < sr[1])]
                                        integrated_counts_denominator += np.sum(sub_data['y_counts_per_seconds'])

                                # Calculate counts for image
                                if 'DivideByIn_' in self.method:
                                    counts_for_image = integrated_counts_numerator / integrated_counts_denominator
                                else:
                                    counts_for_image = integrated_counts_numerator

                            if self.baseline['type'] is not None and (set_pixel_to_zero or counts_for_image < 0):
                                counts_for_image = self.nan_value
                        else:
                            counts_for_image = self.nan_value

                        counts_for_image_along_y.append(counts_for_image)
                    self.image_data_from_spectra.append(counts_for_image_along_y)
                self.image_data_from_spectra = np.transpose(self.image_data_from_spectra)
            else:
                self.type = 'SPCM'
        else:
            self.type = 'SPCM'

        return True

    # This function fits spectral data within chosen spectral range in each pixel and outputs
    # parameters (to a file if chosen)
    def fit_spectra(self,
                    fitting_function='2_voigt_and_linear_baseline',
                    default_widths=0.0001,
                    parameter_scans=3,
                    goodness_of_fit_threshold=0.9,
                    save_to_file=False,
                    file_name_for_fitting_parameter_maps=''):
        def calculate_goodness_of_fit(y_data, y_fit):
            S_res = np.sum((y_data - y_fit) ** 2)
            S_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            return 1 - S_res / S_tot

        def calculate_full_width_at_half_maximum(width_gaussian, width_lorentzian):
            FWHM = []
            for ix in range(len(width_gaussian)):
                FWHM_along_y = []
                for iy in range(len(width_gaussian[0])):
                    if np.isnan(width_gaussian[ix][iy]) or np.isnan(width_lorentzian[ix][iy]):
                        FWHM_along_y.append(np.NaN)
                    else:
                        x = np.linspace(1 - 3 * (width_gaussian[ix][iy] + width_lorentzian[ix][iy]),
                                        1 + 3 * (width_gaussian[ix][iy] + width_lorentzian[ix][iy]), 1000)
                        y = spsp.voigt_profile(x - 1, width_gaussian[ix][iy], width_lorentzian[ix][iy])
                        maximum = np.max(y)
                        df = pd.DataFrame(data={'x': x, 'y': y})
                        df['y_diff'] = np.abs(df.y - maximum / 2)
                        df_left = df[df['x'] < 1]
                        df_right = df[df['x'] > 1]
                        y_diff_left_min = df_left['y_diff'].min()
                        y_diff_right_min = df_right['y_diff'].min()
                        x_left = df_left[df_left['y_diff'] == y_diff_left_min].x.values
                        x_right = df_right[df_right['y_diff'] == y_diff_right_min].x.values
                        FWHM_along_y.append(np.abs(x_left - x_right)[0])
                FWHM.append(FWHM_along_y)
            return FWHM

        def calculate_difference_in_peak_position(peak_position_1, peak_position_2):
            return peak_position_1 - peak_position_2

        def calculate_amplitude_ratio(amplitude_1, amplitude_2):
            return amplitude_1 / amplitude_2

        self.fitting_parameters = {}
        self.fitting_covariance = {}
        self.goodness_of_fit = {}
        self.fitting_parameter_map = {}
        self.positions_of_bad_fits = []
        if fitting_function == '2_voigt_and_linear_baseline':
            fittings_parameter_identifier = ['Amplitude_1', 'Width_Gaussian_1', 'Width_Lorentzian_1', 'Position_Peak_1',
                                             'Amplitude_2', 'Width_Gaussian_2', 'Width_Lorentzian_2', 'Position_Peak_2']
            for identifier in fittings_parameter_identifier:
                self.fitting_parameter_map[identifier] = []
            for ix, x_position in enumerate(self.x):
                fitting_parameter_list = {}
                for identifier in fittings_parameter_identifier:
                    fitting_parameter_list[identifier] = []
                for iy, y_position in enumerate(self.y):
                    position_string = convert_xy_to_position_string([x_position, y_position])
                    self.fitting_parameters[position_string] = None
                    self.fitting_covariance[position_string] = None
                    self.goodness_of_fit[position_string] = 0
                    for n in range(1, parameter_scans + 1):
                        for k in range(1, parameter_scans + 1):
                            try:
                                p0 = np.zeros(10)
                                slope, intersect = guess_linear_fit(self.sub_spectra[position_string],
                                                                    self.unit_spectral_range)
                                p0[0] = slope
                                p0[1] = intersect
                                indexes, _ = sps.find_peaks(
                                    self.sub_spectra[position_string]['y_counts_per_seconds'].values, prominence=(
                                         self.sub_spectra[position_string]['y_counts_per_seconds'].max() -
                                         self.sub_spectra[position_string]['y_counts_per_seconds'].min()) / 10)
                                indexes_local_maxima = self.sub_spectra[position_string].loc[
                                    indexes, 'y_counts_per_seconds'].nlargest(2).index
                                indexes_local_maxima = sorted(indexes_local_maxima)
                                p0[3] = default_widths * n
                                p0[4] = default_widths * n
                                width_voigt = 0.5346 * default_widths + \
                                    np.sqrt(0.2166 * default_widths ** 2 + default_widths ** 2)
                                p0[2] = self.sub_spectra[position_string].loc[
                                            indexes_local_maxima[0], 'y_counts_per_seconds'] * width_voigt * k
                                p0[5] = self.sub_spectra[position_string].loc[
                                    indexes_local_maxima[0], 'x_{0}'.format(self.unit_spectral_range)]
                                p0[7] = default_widths * n
                                p0[8] = default_widths * n
                                width_voigt = 0.5346 * default_widths + \
                                    np.sqrt(0.2166 * default_widths ** 2 + default_widths ** 2)
                                p0[6] = self.sub_spectra[position_string].loc[
                                            indexes_local_maxima[1], 'y_counts_per_seconds'] * width_voigt * k
                                p0[9] = self.sub_spectra[position_string].loc[
                                    indexes_local_maxima[1], 'x_{0}'.format(self.unit_spectral_range)]
                                self.fitting_parameters[position_string], self.fitting_covariance[
                                    position_string] = spo.curve_fit(two_voigt_and_linear_baseline,
                                    self.sub_spectra[position_string]['x_{0}'.format(self.unit_spectral_range)],
                                    self.sub_spectra[position_string]['y_counts_per_seconds'], p0=p0)
                                self.goodness_of_fit[position_string] = calculate_goodness_of_fit(
                                    self.sub_spectra[position_string]['y_counts_per_seconds'],
                                    two_voigt_and_linear_baseline(
                                        self.sub_spectra[position_string]['x_{0}'.format(self.unit_spectral_range)],
                                        self.fitting_parameters[position_string][0],
                                        self.fitting_parameters[position_string][1],
                                        self.fitting_parameters[position_string][2],
                                        self.fitting_parameters[position_string][3],
                                        self.fitting_parameters[position_string][4],
                                        self.fitting_parameters[position_string][5],
                                        self.fitting_parameters[position_string][6],
                                        self.fitting_parameters[position_string][7],
                                        self.fitting_parameters[position_string][8],
                                        self.fitting_parameters[position_string][9]))
                                if self.goodness_of_fit[position_string] >= goodness_of_fit_threshold:
                                    break
                            except RuntimeError:
                                pass
                            except IndexError:
                                break
                        if self.goodness_of_fit[position_string] >= goodness_of_fit_threshold:
                            break
                        if self.goodness_of_fit[position_string] < goodness_of_fit_threshold and n == parameter_scans \
                                and k == parameter_scans:
                            self.positions_of_bad_fits.append(position_string)
                            self.fitting_parameters[position_string] = None
                            self.fitting_covariance[position_string] = None
                    for i, identifier in enumerate(fittings_parameter_identifier):
                        try:
                            fitting_parameter_list[identifier].append(self.fitting_parameters[position_string][i + 2])
                        except TypeError:
                            fitting_parameter_list[identifier].append(np.NaN)
                print('Fitted Line {0}'.format(ix + 1))
                for identifier in fittings_parameter_identifier:
                    self.fitting_parameter_map[identifier].append(fitting_parameter_list[identifier])
            for key in self.fitting_parameter_map:
                self.fitting_parameter_map[key] = np.array(np.transpose(self.fitting_parameter_map[key]))

            self.fitting_parameter_map['FWHM_1'] = calculate_full_width_at_half_maximum(
                self.fitting_parameter_map['Width_Gaussian_1'], self.fitting_parameter_map['Width_Lorentzian_1'])
            self.fitting_parameter_map['FWHM_2'] = calculate_full_width_at_half_maximum(
                self.fitting_parameter_map['Width_Gaussian_2'], self.fitting_parameter_map['Width_Lorentzian_2'])
            self.fitting_parameter_map['Delta_Peak_Positions'] = calculate_difference_in_peak_position(
                self.fitting_parameter_map['Position_Peak_1'], self.fitting_parameter_map['Position_Peak_2'])
            self.fitting_parameter_map['Amplitude_Ratio'] = calculate_amplitude_ratio(
                self.fitting_parameter_map['Amplitude_1'], self.fitting_parameter_map['Amplitude_2'])

        if save_to_file:
            for key in self.fitting_parameter_map:
                file_name_for_fitting_parameter_maps_key = file_name_for_fitting_parameter_maps + '_{0}.txt'.format(key)
                np.savetxt(file_name_for_fitting_parameter_maps_key, self.fitting_parameter_map[key])

        return True

    # Construct image from image data
    def add_image(self,
                  scale_bar=None,  # dict with options (see line 857)
                  scale: Union[str, Dict] = 'Auto',  # dict with options (see line 835)
                  color_bar=True,
                  plot_style=None,
                  image_from_spectra=False,  # Should be true if spectral data were analyzed
                  masking_treshold=None,  # Use masking to not show certain values in image
                  interpolation=None,  # For interpolation used by pcolormesh
                  axis_ticks=False,
                  color_bar_label='Auto',
                  color_bar_ticklabels='Auto',
                  color_bar_ticks='Auto',
                  aspect=None,
                  # The parameters below are for plotting from files created by fit_spectra
                  image_from_fitting_parameters=False, fitting_parameter_identifier='Amplitude_1',
                  image_from_file=False, image_file='', image_identifier='',
                  fig=None,
                  ax=None):
        # Define function for formatting scale bar text
        def format_scale_bar(scale_value, unit='um'):
            if unit == 'um':
                scale_string = r'{0}'.format(int(scale_value)) + r' \textmu m'
            return scale_string

        # Set plotting style
        if plot_style is not None:
            plt.style.use(plot_style)

        # Generate figure, axes
        if fig is None:
            figure = plt.figure(figsize=(15, 10))
        else:
            figure = fig
        if ax is None:
            axes = figure.add_subplot(1, 1, 1)
        else:
            axes = ax

        # Plot image
        if image_from_spectra:
            self.image_data_to_plot = self.image_data_from_spectra
        elif image_from_fitting_parameters:
            self.image_data_to_plot = self.fitting_parameter_map[fitting_parameter_identifier]
        elif image_from_file:
            self.image_data_to_plot = np.loadtxt(image_file)
        else:
            self.image_data_to_plot = self.image_data
        if masking_treshold is not None:
            self.image_data_to_plot = np.ma.masked_where(
                (self.image_data_to_plot <= masking_treshold[0]) | (self.image_data_to_plot >= masking_treshold[1]),
                self.image_data_to_plot)
            
        if isinstance(scale, dict):
            if 'color_map' in scale:
                cmap = plt.get_cmap(scale['color_map'])
                if 'bad' in scale:
                    cmap.set_bad(scale['bad'])
                    
        if scale == 'Auto':
            im = axes.imshow(self.image_data_to_plot,
                             cmap=plt.get_cmap('gray'),
                             extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                     self.extent['y_max']],
                             interpolation=interpolation, aspect=aspect)
        elif scale == 'Normalized':
            im = axes.imshow((self.image_data_to_plot - np.min(self.image_data_to_plot)) / (
                    np.max(self.image_data_to_plot) - np.min(self.image_data_to_plot)),
                             cmap=plt.get_cmap('gray'),
                             extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                     self.extent['y_max']],
                             interpolation=interpolation, aspect=aspect)
        else:
            default_scale = {'minimum_value': np.nanmin(self.image_data_to_plot),
                             'maximum_value': np.nanmax(self.image_data_to_plot), 'norm': None, 'color_map': 'gray'}
            for key in default_scale:
                if key not in scale:
                    scale[key] = default_scale[key]

            if scale['norm'] is None:
                im = axes.imshow(self.image_data_to_plot,
                                 cmap=cmap,
                                 extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                         self.extent['y_max']],
                                 vmin=scale['minimum_value'], vmax=scale['maximum_value'],
                                 interpolation=interpolation, aspect=aspect)
            elif scale['norm'] == 'log':
                im = axes.imshow(self.image_data_to_plot,
                                 cmap=cmap,
                                 extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                         self.extent['y_max']],
                                 norm=LogNorm(vmin=scale['minimum_value'], vmax=scale['maximum_value']),
                                 interpolation=interpolation, aspect=aspect)

        # Add scale bar
        if scale_bar is not None:
            default_scale_bar = {'scale': 5, 'font_size': 24, 'color': 'white', 'position': 'lower left',
                                 'scale_adjust_factor': 1}
            if not isinstance(scale_bar, dict):
                default_scale_bar['scale'] = scale_bar
                scale_bar = default_scale_bar
            else:
                for key in default_scale_bar:
                    if key not in scale_bar:
                        scale_bar[key] = default_scale_bar[key]
            fontprops = fm.FontProperties(size=scale_bar['font_size'])
            scalebar = AnchoredSizeBar(axes.transData,
                                       scale_bar['scale'] * scale_bar['scale_adjust_factor'],
                                       format_scale_bar(scale_bar['scale']),
                                       scale_bar['position'],
                                       pad=0.1,
                                       color=scale_bar['color'],
                                       frameon=False,
                                       size_vertical=0.5,
                                       sep=5,
                                       fontproperties=fontprops)

            axes.add_artist(scalebar)

        # Turn axes labels/ticks off
        if axis_ticks:
            axes.set_xlabel(r'Location $x$ ($\mathrm{\mu}$m)')
            axes.set_ylabel(r'Location $y$ ($\mathrm{\mu}$m)')
        else:
            plt.axis('off')

        # Display color bar
        if color_bar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if color_bar_ticks == 'Auto':
                cbar = figure.colorbar(im, cax=cax, orientation='vertical')
            else:
                cbar = figure.colorbar(im, cax=cax, orientation='vertical', ticks=color_bar_ticks)
            if color_bar_label == 'Auto':
                if (not image_from_fitting_parameters) and (not image_from_file):
                    if self.method == 'sum':
                        if scale == 'Normalized':
                            cax.set_ylabel('Normalized PL Intensity (A.U.)')
                        else:
                            cax.set_ylabel('PL Intensity (counts/second)')
                    elif 'position' in self.method:
                        if self.unit_spectral_range == 'eV':
                            cax.set_ylabel('Photon Energy (eV)')
                        elif self.unit_spectral_range == 'nm':
                            cax.set_ylabel('Wavelength (nm)')
                    elif 'intensity' in self.method:
                        if 'ratio' in self.method:
                            cax.set_ylabel('PL Intensity Ratio (A.U.)')
                        else:
                            if scale == 'Normalized':
                                cax.set_ylabel('Normalized PL Intensity (A.U.)')
                            else:
                                cax.set_ylabel('PL Intensity (A.U.)')
                elif not image_from_file:
                    if fitting_parameter_identifier in ['Width_Gaussian_1', 'Width_Lorentzian_1', 'Position_Peak_1',
                                                        'Width_Gaussian_2', 'Width_Lorentzian_2', 'Position_Peak_2',
                                                        'FWHM_1', 'FWHM_2', 'Delta_Peak_Positions']:
                        if self.unit_spectral_range == 'eV':
                            cax.set_ylabel('Photon Energy (eV)')
                        elif self.unit_spectral_range == 'nm':
                            cax.set_ylabel('Wavelength (nm)')
                    elif fitting_parameter_identifier in ['Amplitude_1', 'Amplitude_2', 'Amplitude_Ratio']:
                        if scale == 'Normalized':
                            cax.set_ylabel('Normalized PL Intensity (A.U.)')
                        else:
                            cax.set_ylabel('PL Intensity (A.U.)')
                else:
                    if image_identifier in ['Width_Gaussian_1', 'Width_Lorentzian_1', 'Position_Peak_1',
                                            'Width_Gaussian_2', 'Width_Lorentzian_2', 'Position_Peak_2', 'FWHM_1',
                                            'FWHM_2', 'Delta_Peak_Positions']:
                        if self.unit_spectral_range == 'eV':
                            cax.set_ylabel('Photon Energy (eV)')
                        elif self.unit_spectral_range == 'nm':
                            cax.set_ylabel('Wavelength (nm)')
                    elif image_identifier in ['Amplitude_1', 'Amplitude_2', 'Amplitude_Ratio']:
                        if scale == 'Normalized':
                            cax.set_ylabel('Normalized PL Intensity (A.U.)')
                        else:
                            cax.set_ylabel('PL Intensity (A.U.)')
            else:
                cax.set_ylabel(color_bar_label)

        # Set nan pixel to red
        color_map = mpl.cm.get_cmap()
        color_map.set_bad(color='white', alpha=1)

        if color_bar_ticklabels != 'Auto':
            cax.set_yticklabels(color_bar_ticklabels)

        # Add figure and axes for to self further manipulation
        self.image = {'figure': figure, 'axes': axes}

        return True

    # Define function for adding markers on image
    def add_marker_to_image(self, marker=None):
        def convert_marker_string_to_marker_array(string):
            string_components = string.split('_')
            axes = ['x', 'y']
            marker = []
            for string_component in string_components:
                for axis in axes:
                    if axis in string_component:
                        number_string = string_component.split(axis)[1]
                        number = convert_to_string_to_float(number_string)
                        break
                marker.append(number)

            if len(marker) != 2:
                raise ValueError('String entered for marker is not in correct format.')

            return marker

        marker_default = {'type': 'dot', 'color': 'red', 'x_position': 0, 'y_position': 0, 'width': 0, 'height': 0,
                          'size': 100, 'position_string': None}
        # Test if self.image exists
        try:
            if not isinstance(marker, dict):
                if isinstance(marker, str):
                    marker = convert_marker_string_to_marker_array(marker)
                marker = {'x_position': marker[0], 'y_position': marker[1]}
            for key in marker_default:
                if key not in marker:
                    marker[key] = marker_default[key]

            # Overwrite x and y position with position_string if it was given
            if marker['position_string'] is not None:
                x_position, y_position = convert_marker_string_to_marker_array(marker['position_string'])
                marker['x_position'] = x_position
                marker['y_position'] = y_position

            # Add marker point
            if marker['type'] == 'dot':
                self.image['axes'].scatter(marker['x_position'], marker['y_position'], edgecolors='face',
                                           c=marker['color'], s=marker['size'], alpha=0.5)
            elif marker['type'] == 'area':
                if marker['size'] == 100:
                    marker['size'] = 3
                area = patches.Rectangle((marker['x_position'], marker['y_position']), marker['width'],
                                         marker['height'],
                                         linewidth=marker['size'], edgecolor=marker['color'], facecolor='none', alpha=0.5)
                self.image['axes'].add_patch(area)
        except NameError:
            raise NameError('Image was not added yet. Run .add_image() first!')

        return True

    # Add histogram
    def add_histogram(self, image_from_spectra=False, plot_style=None, bins='auto'):
        # Set plotting style
        if plot_style is not None:
            plt.style.use(plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

        axes.set_ylabel('Frequency')
        if self.method == 'sum':
            axes.set_xlabel('PL Intensity (counts/second)')
        elif 'position' in self.method:
            if self.unit_spectral_range == 'eV':
                axes.set_xlabel('Photon Energy (eV)')
            elif self.unit_spectral_range == 'nm':
                axes.set_xlabel('Wavelength (nm)')

        # Plot image
        if image_from_spectra:
            self.histogram_data = self.image_data_from_spectra.flatten()
        else:
            self.histogram_data = self.image_data.flatten()

        if bins == 'auto':
            bins = int(np.sqrt(len(self.histogram_data)))

        axes.hist(self.histogram_data, bins=bins)

        # Add figure and axes for to self further manipulation
        self.histogram = {'figure': figure, 'axes': axes}

        return True

    # Save image
    def save_image(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.image['figure'].savefig(title, bbox_inches='tight', transparent=True)

        return True

    # Save histogram
    def save_histogram(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.histogram['figure'].savefig(title, bbox_inches='tight', transparent=True)

        return True
