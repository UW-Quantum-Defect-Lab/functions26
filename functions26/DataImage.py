# 2021-01-27
# This code was made for use in the Fu lab
# by Christian Zimmermann

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
import warnings

from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .constants import conversion_factor_nm_to_ev  # eV*nm
from .constants import n_air
from .DataDictXXX import DataDictFilenameInfo


def convert_to_string_to_float(number_string):
    # Figure out sign
    if 'n' in number_string:
        sign = -1
        number_string = number_string.split('n')[1]
    elif 'm' in number_string:
        sign = -1
        number_string = number_string.split('m')[1]
    elif '-' in number_string:
        sign = -1
        number_string = number_string.split('-')[1]
    else:
        sign = 1

    # Figure out decimal point
    if 'p' in number_string:
        number = float(number_string.replace('p', '.'))
    else:
        number = float(number_string)

    # Apply sign
    number *= sign

    return number


def convert_xy_to_position_string(position):
    axes = ['x', 'y']
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


def line(x, a, b):
    return a*x+b


def exponential(x, a, b, x0):
    return a*np.exp(b*(x-x0))


def guess_initial_parameters(x, y, function = 'linear'):
    index_min = x.idxmin()
    index_max = x.idxmax()
    x0 = x.loc[index_min]
    x1 = x.loc[index_max]
    y0 = y.loc[index_min]
    y1 = y.loc[index_max]
    if function == 'linear':
        slope = (y1-y0)/(x1-x0)
        intersect = y1 - slope*x1
        p0 = [slope, intersect]
    elif function == 'exponential':
        exponent = (np.log(y1) - np.log(y0))/(x1 - x0)
        prefactor = y0
        shift = x0
        p0 = [prefactor, exponent, shift]

    return p0


def two_voigt_and_linear_baseline(x, slope, intersect, amplitude_1, width_gaussian_1, width_lorentzian_1, position_1, amplitude_2, width_gaussian_2, width_lorentzian_2, position_2):
    return (line(x, slope, intersect)
                + amplitude_1*spsp.voigt_profile(x - position_1, width_gaussian_1, width_lorentzian_1)
                + amplitude_2*spsp.voigt_profile(x - position_2, width_gaussian_2, width_lorentzian_2))


def guess_linear_fit(sub_spectrum, unit_spectral_range, number_of_points = 10):
    sub_spectrum.reset_index(drop = True, inplace = True)
    sub_spectrum_left = sub_spectrum.loc[0:number_of_points]
    sub_spectrum_right = sub_spectrum.loc[len(sub_spectrum.index)-number_of_points+1:len(sub_spectrum.index)+1]
    sub_spectrum_left.reset_index(drop = True, inplace = True)
    sub_spectrum_right.reset_index(drop = True, inplace = True)
    slope = (sub_spectrum_right['y_counts_per_seconds'][0] - sub_spectrum_left['y_counts_per_seconds'][0])/(sub_spectrum_right['x_{0}'.format(unit_spectral_range)][0] - sub_spectrum_left['x_{0}'.format(unit_spectral_range)][0])
    intersect = sub_spectrum_left['y_counts_per_seconds'][0] - slope*sub_spectrum_left['x_{0}'.format(unit_spectral_range)][0]
    sub_spectrum_edges = pd.concat([sub_spectrum_left, sub_spectrum_right], ignore_index = True)
    params, covar = spo.curve_fit(line, sub_spectrum_edges['y_counts_per_seconds'], sub_spectrum_edges['x_{0}'.format(unit_spectral_range)], p0 = [slope, intersect])
    slope = params[0]
    intersect = params[1]
    return slope, intersect


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
        self.extent = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
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


class DataConfocalScan(DataImage):
    allowed_file_extensions = ['mat']

    def __init__(self, file_name, folder_name='.', spectral_range='all', unit_spectral_range=None, baseline=None,
                 method='sum', background=300, wavelength_offset=0, new_wavelength_axis=None, second_order=True,
                 refractive_index=n_air):

        self.spectral_range = spectral_range
        self.unit_spectral_range = unit_spectral_range
        self.background = background
        self.second_order = second_order
        self.refractive_index = refractive_index
        self.wavelength_offset = wavelength_offset
        self.new_wavelength_axis = new_wavelength_axis
        self.method = method
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

    def get_data(self):

        matlab_file_data = spio.loadmat(self.file_name)

        if 'scan' in matlab_file_data.keys():
            self.software = 'DoritoScopeConfocal'
            self.image_data = matlab_file_data['scan'][0][0][4]
            self.exposure_time = matlab_file_data['scan'][0][0][11][0][0]
            self.image_data = self.image_data/self.exposure_time
            # Convert image, so it looks like what we see in the matlab GUI
            self.image_data = np.transpose(self.image_data)
            self.image_data = np.flip(self.image_data, axis = 0)

            self.x = matlab_file_data['scan'][0][0][0][0]
            self.y = matlab_file_data['scan'][0][0][1][0]
            self.y = np.flip(self.y)
        elif 'data' in matlab_file_data.keys():
            self.software = 'McDiamond'
            self.image_data = matlab_file_data['data'][0][0][7][0][0]
            # Convert image, so it looks like what we see in the matlab GUI
            self.image_data = np.transpose(self.image_data)
            self.image_data = np.flip(self.image_data, axis = 0)

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
                self.spectra_raw = self.spectra_raw/self.exposure_time
                if self.new_wavelength_axis is not None:
                    self.wavelength = self.new_wavelength_axis + self.wavelength_offset
                else:
                    self.wavelength = matlab_file_data['scan'][0][0][16][0] + self.wavelength_offset
                if self.second_order:
                    self.wavelength = self.wavelength / 2
                self.photon_energy = conversion_factor_nm_to_ev /(self.wavelength*self.refractive_index)
                self.spectra = {}
                for ix, x_position in enumerate(self.x):
                    for iy, y_position in enumerate(self.y):
                        position_string = convert_xy_to_position_string([x_position, y_position])
                        self.spectra[position_string] = pd.DataFrame(
                            data={'x_nm': self.wavelength, 'y_counts_per_seconds': self.spectra_raw[iy][ix]})
                        self.spectra[position_string]['x_eV'] = self.photon_energy

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
                            if self.baseline['type'] != None:
                                self.baseline[position_string] = {}
                                if self.baseline['method_left'] == 'edge':
                                    index_left = np.abs(spectrum['x_{0}'.format(self.unit_spectral_range)] - self.spectral_range[0]).idxmin()
                                elif self.baseline['method_left'] == 'minimum':
                                    index_left = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                                    spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]['y_counts_per_seconds'].idxmin()
                                elif self.baseline['method_left'] == 'maximum':
                                    index_left = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                                    spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]['y_counts_per_seconds'].idxmax()
                                if self.baseline['method_right'] == 'edge':
                                    index_right = np.abs(spectrum['x_{0}'.format(self.unit_spectral_range)] - self.spectral_range[1]).idxmin()
                                elif self.baseline['method_right'] == 'minimum':
                                    index_right = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                                    spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]['y_counts_per_seconds'].idxmin()
                                elif self.baseline['method_right'] == 'maximum':
                                    index_right = spectrum.loc[(spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                                    spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]['y_counts_per_seconds'].idxmax()
                                sub_spectrum_for_fitting_left = spectrum.loc[index_left - self.baseline['points_left'] : index_left + self.baseline['points_left']]
                                sub_spectrum_for_fitting_right = spectrum.loc[index_right - self.baseline['points_right'] : index_right + self.baseline['points_right']]
                                if 'bi' not in self.baseline['type']:
                                    self.baseline[position_string]['sub_spectrum_for_fitting'] = pd.concat([sub_spectrum_for_fitting_left, sub_spectrum_for_fitting_right], ignore_index = True)
                                self.baseline[position_string]['sub_spectrum_for_fitting_left'] = sub_spectrum_for_fitting_left
                                self.baseline[position_string]['sub_spectrum_for_fitting_right'] = sub_spectrum_for_fitting_right
                                try:
                                    set_pixel_to_zero = False
                                    if self.baseline['type'] == 'linear':
                                        p0 = guess_initial_parameters(self.baseline[position_string]['sub_spectrum_for_fitting']['x_{0}'.format(self.unit_spectral_range)],
                                                                                     self.baseline[position_string]['sub_spectrum_for_fitting']['y_counts_per_seconds'], 'linear')
                                        parameters, covariance = spo.curve_fit(line, self.baseline[position_string]['sub_spectrum_for_fitting']['x_{0}'.format(self.unit_spectral_range)],
                                                                                     self.baseline[position_string]['sub_spectrum_for_fitting']['y_counts_per_seconds'], p0 = p0)
                                        self.baseline[position_string]['slope_initial'] = p0[0]
                                        self.baseline[position_string]['intersect_initial'] = p0[1]
                                        self.baseline[position_string]['slope'] = parameters[0]
                                        self.baseline[position_string]['intersect'] = parameters[1]
                                    elif self.baseline['type'] == 'minimum':
                                        self.baseline[position_string]['offset'] = spectrum['y_counts_per_seconds'].min()
                                    elif self.baseline['type'] == 'exponential':
                                        p0 = guess_initial_parameters(self.baseline[position_string]['sub_spectrum_for_fitting']['x_{0}'.format(self.unit_spectral_range)],
                                                                                     self.baseline[position_string]['sub_spectrum_for_fitting']['y_counts_per_seconds'], 'exponential')
                                        parameters, covariance = spo.curve_fit(exponential, self.baseline[position_string]['sub_spectrum_for_fitting']['x_{0}'.format(self.unit_spectral_range)],
                                                                                     self.baseline[position_string]['sub_spectrum_for_fitting']['y_counts_per_seconds'], p0 = p0)
                                        self.baseline[position_string]['prefactor_initial'] = p0[0]
                                        self.baseline[position_string]['exponent_initial'] = p0[1]
                                        self.baseline[position_string]['shift_initial'] = p0[2]
                                        self.baseline[position_string]['prefactor'] = parameters[0]
                                        self.baseline[position_string]['exponent'] = parameters[1]
                                        self.baseline[position_string]['shift'] = parameters[2]
                                    elif self.baseline['type'] == 'bilinear':
                                        p0 = guess_initial_parameters(sub_spectrum_for_fitting_left['x_{0}'.format(self.unit_spectral_range)], sub_spectrum_for_fitting_left['y_counts_per_seconds'], 'linear')
                                        parameters, covariance = spo.curve_fit(line, sub_spectrum_for_fitting_left['x_{0}'.format(self.unit_spectral_range)], sub_spectrum_for_fitting_left['y_counts_per_seconds'], p0 = p0)
                                        self.baseline[position_string]['slope_initial_left'] = p0[0]
                                        self.baseline[position_string]['intersect_initial_left'] = p0[1]
                                        self.baseline[position_string]['slope_left'] = parameters[0]
                                        self.baseline[position_string]['intersect_left'] = parameters[1]
                                        p0 = guess_initial_parameters(sub_spectrum_for_fitting_right['x_{0}'.format(self.unit_spectral_range)], sub_spectrum_for_fitting_right['y_counts_per_seconds'], 'linear')
                                        parameters, covariance = spo.curve_fit(line, sub_spectrum_for_fitting_right['x_{0}'.format(self.unit_spectral_range)], sub_spectrum_for_fitting_right['y_counts_per_seconds'], p0 = p0)
                                        self.baseline[position_string]['slope_initial_right'] = p0[0]
                                        self.baseline[position_string]['intersect_initial_right'] = p0[1]
                                        self.baseline[position_string]['slope_right'] = parameters[0]
                                        self.baseline[position_string]['intersect_right'] = parameters[1]
                                except RuntimeError:
                                    set_pixel_to_zero = True
                            spectrum = spectrum.loc[
                                (spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                            spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]
                            self.sub_spectra[position_string] = spectrum

                            if self.baseline['type'] != None and not set_pixel_to_zero:
                                self.baseline[position_string]['x'] = spectrum['x_{0}'.format(self.unit_spectral_range)]
                                if self.baseline['type']== 'linear':
                                    self.baseline[position_string]['y'] = line(self.baseline[position_string]['x'], self.baseline[position_string]['slope'], self.baseline[position_string]['intersect'])
                                    self.baseline[position_string]['y_initial'] = line(self.baseline[position_string]['x'], self.baseline[position_string]['slope_initial'], self.baseline[position_string]['intersect_initial'])
                                elif self.baseline['type'] == 'exponential':
                                    self.baseline[position_string]['y'] = exponential(self.baseline[position_string]['x'], self.baseline[position_string]['prefactor'], self.baseline[position_string]['exponent'], self.baseline[position_string]['shift'])
                                    self.baseline[position_string]['y_initial'] = exponential(self.baseline[position_string]['x'], self.baseline[position_string]['prefactor_initial'], self.baseline[position_string]['exponent_initial'], self.baseline[position_string]['shift_initial'])
                                elif self.baseline['type'] == 'minimum':
                                    self.baseline[position_string]['y'] = line(self.baseline[position_string]['x'], 0, self.baseline[position_string]['offset'])
                                elif self.baseline['type'] == 'bilinear':
                                    line_left = line(self.baseline[position_string]['x'], self.baseline[position_string]['slope_left'], self.baseline[position_string]['intersect_left'])
                                    line_right = line(self.baseline[position_string]['x'], self.baseline[position_string]['slope_right'], self.baseline[position_string]['intersect_right'])
                                    self.baseline[position_string]['y'] = np.maximum(line_left, line_right)
                                    line_left_initial = line(self.baseline[position_string]['x'], self.baseline[position_string]['slope_initial_left'], self.baseline[position_string]['intersect_initial_left'])
                                    line_right_initial = line(self.baseline[position_string]['x'], self.baseline[position_string]['slope_initial_right'], self.baseline[position_string]['intersect_initial_right'])
                                    self.baseline[position_string]['y_initial'] = np.maximum(line_left, line_right)
                                spectrum['y_counts_per_seconds'] = spectrum['y_counts_per_seconds'] - self.baseline[position_string]['y']

                        if self.method == 'sum':
                            counts_for_image = spectrum['y_counts_per_seconds'].sum()
                        elif self.method == 'maximum_position':
                            index = spectrum['y_counts_per_seconds'].idxmax()
                            counts_for_image = spectrum.loc[index, 'x_{0}'.format(self.unit_spectral_range)]
                        elif self.method == 'center_of_mass_position':
                            weights = np.maximum(spectrum['y_counts_per_seconds'], 0)
                            counts_for_image = np.average(spectrum['x_{0}'.format(self.unit_spectral_range)].to_numpy(), weights = weights)
                        elif 'local_maximum' in self.method:
                            spectrum.reset_index(drop = True, inplace = True)
                            indexes, _ = sps.find_peaks(spectrum['y_counts_per_seconds'].values, prominence = (spectrum['y_counts_per_seconds'].max()-spectrum['y_counts_per_seconds'].min())/10)
                            number_of_peaks = int(self.method.split('_')[3])
                            indexes = spectrum.loc[indexes, 'y_counts_per_seconds'].nlargest(number_of_peaks).index
                            indexes = sorted(indexes)
                            number = int(self.method.split('_')[4]) - 1
                            if 'position' in self.method:
                                try:
                                    counts_for_image = spectrum.loc[indexes[number], 'x_{0}'.format(self.unit_spectral_range)]
                                except IndexError:
                                    counts_for_image = np.NaN
                            elif 'intensity' in self.method:
                                try:
                                    counts_for_image = spectrum.loc[indexes[number], 'y_counts_per_seconds']
                                except IndexError:
                                    counts_for_image = np.NaN
                        elif 'local_maxima_intensity_ratio' in self.method:
                            spectrum.reset_index(drop = True, inplace = True)
                            indexes, _ = sps.find_peaks(spectrum['y_counts_per_seconds'].values, prominence = (spectrum['y_counts_per_seconds'].max()-spectrum['y_counts_per_seconds'].min())/10)
                            indexes = spectrum.loc[indexes, 'y_counts_per_seconds'].nlargest(2).index
                            indexes = sorted(indexes)
                            if 'subtract_minimum' in self.method:
                                baseline = np.nanmin(spectrum['y_counts_per_seconds'].values)
                            else:
                                baseline = 0
                            try:
                                counts_for_image = (spectrum.loc[indexes[0], 'y_counts_per_seconds'] - baseline)/(spectrum.loc[indexes[1], 'y_counts_per_seconds'] - baseline)
                            except IndexError:
                                counts_for_image = np.NaN

                        if self.baseline['type'] != None and (set_pixel_to_zero or counts_for_image < 0):
                            counts_for_image = np.NaN

                        counts_for_image_along_y.append(counts_for_image)
                    self.image_data_from_spectra.append(counts_for_image_along_y)
                self.image_data_from_spectra = np.transpose(self.image_data_from_spectra)
            else:
                self.type = 'SPCM'
        else:
            self.type = 'SPCM'

        return True

    def fit_spectra(self, fitting_function = '2_voigt_and_linear_baseline', default_widths = 0.0001, parameter_scans = 3, goodness_of_fit_threshold = 0.9, save_to_file = False, file_name_for_fitting_parameter_maps = ''):
        def calculate_goodness_of_fit(y_data, y_fit):
            S_res = np.sum((y_data - y_fit)**2)
            S_tot = np.sum((y_data - np.mean(y_data))**2)
            return 1 - S_res/S_tot

        def calculate_full_width_at_half_maximum(width_gaussian, width_lorentzian):
            FWHM = []
            for ix in range(len(width_gaussian)):
                FWHM_along_y = []
                for iy in range(len(width_gaussian[0])):
                    if np.isnan(width_gaussian[ix][iy]) or np.isnan(width_lorentzian[ix][iy]):
                        FWHM_along_y.append(np.NaN)
                    else:
                        x = np.linspace(1-3*(width_gaussian[ix][iy]+width_lorentzian[ix][iy]), 1+3*(width_gaussian[ix][iy]+width_lorentzian[ix][iy]), 1000)
                        y = spsp.voigt_profile(x - 1, width_gaussian[ix][iy], width_lorentzian[ix][iy])
                        maximum = np.max(y)
                        df = pd.DataFrame(data = {'x' : x, 'y' : y})
                        df['y_diff'] = np.abs(df.y - maximum/2)
                        df_left = df[df['x'] < 1]
                        df_right = df[df['x'] > 1]
                        y_diff_left_min =  df_left['y_diff'].min()
                        y_diff_right_min =  df_right['y_diff'].min()
                        x_left = df_left[df_left['y_diff'] == y_diff_left_min].x.values
                        x_right = df_right[df_right['y_diff'] == y_diff_right_min].x.values
                        FWHM_along_y.append(np.abs(x_left - x_right)[0])
                FWHM.append(FWHM_along_y)
            return FWHM

        def calculate_difference_in_peak_position(peak_position_1, peak_position_2):
            return peak_position_1 - peak_position_2

        def calculate_amplitude_ratio(amplitude_1, amplitude_2):
            return amplitude_1/amplitude_2

        self.fitting_parameters = {}
        self.fitting_covariance = {}
        self.goodness_of_fit = {}
        self.fitting_parameter_map = {}
        self.positions_of_bad_fits = []
        if fitting_function == '2_voigt_and_linear_baseline':
            fittings_parameter_identifier = ['Amplitude_1', 'Width_Gaussian_1', 'Width_Lorentzian_1', 'Position_Peak_1', 'Amplitude_2', 'Width_Gaussian_2', 'Width_Lorentzian_2', 'Position_Peak_2']
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
                    for n in range(1,parameter_scans + 1):
                        for k in range(1,parameter_scans + 1):
                            try:
                                p0 = np.zeros(10)
                                slope, intersect = guess_linear_fit(self.sub_spectra[position_string], self.unit_spectral_range)
                                p0[0] = slope
                                p0[1] = intersect
                                indexes, _ = sps.find_peaks(self.sub_spectra[position_string]['y_counts_per_seconds'].values, prominence = (self.sub_spectra[position_string]['y_counts_per_seconds'].max()-self.sub_spectra[position_string]['y_counts_per_seconds'].min())/10)
                                indexes_local_maxima = self.sub_spectra[position_string].loc[indexes, 'y_counts_per_seconds'].nlargest(2).index
                                indexes_local_maxima = sorted(indexes_local_maxima)
                                p0[3] = default_widths*n
                                p0[4] = default_widths*n
                                width_voigt = 0.5346*default_widths + np.sqrt(0.2166*default_widths**2 + default_widths**2)
                                p0[2] = self.sub_spectra[position_string].loc[indexes_local_maxima[0], 'y_counts_per_seconds']*width_voigt*k
                                p0[5] = self.sub_spectra[position_string].loc[indexes_local_maxima[0], 'x_{0}'.format(self.unit_spectral_range)]
                                p0[7] = default_widths*n
                                p0[8] = default_widths*n
                                width_voigt = 0.5346*default_widths + np.sqrt(0.2166*default_widths**2 + default_widths**2)
                                p0[6] = self.sub_spectra[position_string].loc[indexes_local_maxima[1], 'y_counts_per_seconds']*width_voigt*k
                                p0[9] = self.sub_spectra[position_string].loc[indexes_local_maxima[1], 'x_{0}'.format(self.unit_spectral_range)]
                                self.fitting_parameters[position_string], self.fitting_covariance[position_string] = spo.curve_fit(two_voigt_and_linear_baseline,
                                                                        self.sub_spectra[position_string]['x_{0}'.format(self.unit_spectral_range)], self.sub_spectra[position_string]['y_counts_per_seconds'],
                                                                        p0 = p0)
                                self.goodness_of_fit[position_string] = calculate_goodness_of_fit(self.sub_spectra[position_string]['y_counts_per_seconds'],
                                                                                two_voigt_and_linear_baseline(self.sub_spectra[position_string]['x_{0}'.format(self.unit_spectral_range)],
                                                                                self.fitting_parameters[position_string][0], self.fitting_parameters[position_string][1],
                                                                                self.fitting_parameters[position_string][2], self.fitting_parameters[position_string][3], self.fitting_parameters[position_string][4], self.fitting_parameters[position_string][5],
                                                                                self.fitting_parameters[position_string][6], self.fitting_parameters[position_string][7], self.fitting_parameters[position_string][8], self.fitting_parameters[position_string][9]))
                                if self.goodness_of_fit[position_string] >= goodness_of_fit_threshold:
                                    break
                            except RuntimeError:
                                pass
                            except IndexError:
                                break
                        if self.goodness_of_fit[position_string] >= goodness_of_fit_threshold:
                            break
                        if self.goodness_of_fit[position_string] < goodness_of_fit_threshold and n == parameter_scans and k == parameter_scans:
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

            self.fitting_parameter_map['FWHM_1'] = calculate_full_width_at_half_maximum(self.fitting_parameter_map['Width_Gaussian_1'], self.fitting_parameter_map['Width_Lorentzian_1'])
            self.fitting_parameter_map['FWHM_2'] = calculate_full_width_at_half_maximum(self.fitting_parameter_map['Width_Gaussian_2'], self.fitting_parameter_map['Width_Lorentzian_2'])
            self.fitting_parameter_map['Delta_Peak_Positions'] = calculate_difference_in_peak_position(self.fitting_parameter_map['Position_Peak_1'], self.fitting_parameter_map['Position_Peak_2'])
            self.fitting_parameter_map['Amplitude_Ratio'] = calculate_amplitude_ratio(self.fitting_parameter_map['Amplitude_1'], self.fitting_parameter_map['Amplitude_2'])

        if save_to_file:
            for key in self.fitting_parameter_map:
                file_name_for_fitting_parameter_maps_key = file_name_for_fitting_parameter_maps + '_{0}.txt'.format(key)
                np.savetxt(file_name_for_fitting_parameter_maps_key, self.fitting_parameter_map[key])

        return True

    def add_image(self, scale_bar=None, scale='Auto', color_bar=True, plot_style=None, image_from_spectra=False,
                  masking_treshold = None,
                  interpolation=None,
                  axis_ticks = False,
                  image_from_fitting_parameters = False, fitting_parameter_identifier = 'Amplitude_1',
                  image_from_file = False, image_file = '', image_identifier = ''):
        # Define function for formatting scale bar text
        def format_scale_bar(scale_value, unit='um'):
            if unit == 'um':
                scale_string = r'{0}'.format(int(scale_value)) + r' $\mathrm{\mu}$m'
            return scale_string

        # Set plotting style
        if plot_style is not None:
            plt.style.use(plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

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
            self.image_data_to_plot = np.ma.masked_where((self.image_data_to_plot <= masking_treshold[0]) | (self.image_data_to_plot >= masking_treshold[1]),
                                                         self.image_data_to_plot)
        if scale == 'Auto':
            im = axes.imshow(self.image_data_to_plot,
                             cmap=plt.get_cmap('gray'),
                             extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                     self.extent['y_max']],
                             interpolation=interpolation)
        elif scale == 'Normalized':
            im = axes.imshow((self.image_data_to_plot - np.min(self.image_data_to_plot)) / (
                        np.max(self.image_data_to_plot) - np.min(self.image_data_to_plot)),
                             cmap=plt.get_cmap('gray'),
                             extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                     self.extent['y_max']],
                             interpolation=interpolation)
        else:
            default_scale = {'minimum_value': np.nanmin(self.image_data_to_plot), 'maximum_value': np.nanmax(self.image_data_to_plot), 'norm': None, 'color_map': 'gray'}
            for key in default_scale:
                if key not in scale:
                    scale[key] = default_scale[key]

            if scale['norm'] is None:
                im = axes.imshow(self.image_data_to_plot,
                                 cmap=plt.get_cmap(scale['color_map']),
                                 extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                         self.extent['y_max']],
                                 vmin=scale['minimum_value'], vmax=scale['maximum_value'],
                                 interpolation=interpolation)
            elif scale['norm'] == 'log':
                im = axes.imshow(self.image_data_to_plot,
                                 cmap=plt.get_cmap(scale['color_map']),
                                 extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                         self.extent['y_max']],
                                 norm=LogNorm(vmin=scale['minimum_value'], vmax=scale['maximum_value']),
                                 interpolation=interpolation)

        # Add scale bar
        if scale_bar is not None:
            default_scale_bar = {'scale': 5, 'font_size': 24, 'color': 'white', 'position': 'lower left'}
            if not isinstance(scale_bar, dict):
                default_scale_bar['scale'] = scale_bar
                scale_bar = default_scale_bar
            else:
                for key in default_scale_bar:
                    if key not in scale_bar:
                        scale_bar[key] = default_scale_bar[key]
            fontprops = fm.FontProperties(size=scale_bar['font_size'])
            scalebar = AnchoredSizeBar(axes.transData,
                                       scale_bar['scale'],
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
            cbar = figure.colorbar(im, cax=cax, orientation='vertical')
            if (not image_from_fitting_parameters) and (not image_from_file):
                if self.method == 'sum':
                    if scale == 'Normalized':
                        cax.set_ylabel('Normalized PL-Intensity (rel. units)')
                    else:
                        cax.set_ylabel('PL-Intensity (counts/second)')
                elif 'position' in self.method:
                    if self.unit_spectral_range == 'eV':
                        cax.set_ylabel('Photon Energy (eV)')
                    elif self.unit_spectral_range == 'nm':
                        cax.set_ylabel('Wavelength (nm)')
                elif 'intensity' in self.method:
                    if 'ratio' in self.method:
                        cax.set_ylabel('PL-Intensity Ratio (rel. units)')
                    else:
                        if scale == 'Normalized':
                            cax.set_ylabel('Normalized PL-Intensity (rel. units)')
                        else:
                            cax.set_ylabel('PL-Intensity (rel. units)')
            elif (not image_from_file):
                if fitting_parameter_identifier in ['Width_Gaussian_1', 'Width_Lorentzian_1', 'Position_Peak_1', 'Width_Gaussian_2', 'Width_Lorentzian_2', 'Position_Peak_2', 'FWHM_1', 'FWHM_2', 'Delta_Peak_Positions']:
                    if self.unit_spectral_range == 'eV':
                        cax.set_ylabel('Photon Energy (eV)')
                    elif self.unit_spectral_range == 'nm':
                        cax.set_ylabel('Wavelength (nm)')
                elif fitting_parameter_identifier in ['Amplitude_1', 'Amplitude_2', 'Amplitude_Ratio']:
                    if scale == 'Normalized':
                        cax.set_ylabel('Normalized PL-Intensity (rel. units)')
                    else:
                        cax.set_ylabel('PL-Intensity (rel. units)')
            else:
                if image_identifier in ['Width_Gaussian_1', 'Width_Lorentzian_1', 'Position_Peak_1', 'Width_Gaussian_2', 'Width_Lorentzian_2', 'Position_Peak_2', 'FWHM_1', 'FWHM_2', 'Delta_Peak_Positions']:
                    if self.unit_spectral_range == 'eV':
                        cax.set_ylabel('Photon Energy (eV)')
                    elif self.unit_spectral_range == 'nm':
                        cax.set_ylabel('Wavelength (nm)')
                elif image_identifier in ['Amplitude_1', 'Amplitude_2', 'Amplitude_Ratio']:
                    if scale == 'Normalized':
                        cax.set_ylabel('Normalized PL-Intensity (rel. units)')
                    else:
                        cax.set_ylabel('PL-Intensity (rel. units)')

        # Set nan pixel to red
        color_map = mpl.cm.get_cmap()
        color_map.set_bad(color = 'white', alpha = 0)


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
                                           c=marker['color'], s=marker['size'])
            elif marker['type'] == 'area':
                if marker['size'] == 100:
                    marker['size'] = 3
                area = patches.Rectangle((marker['x_position'], marker['y_position']), marker['width'],
                                         marker['height'],
                                         linewidth=marker['size'], edgecolor=marker['color'], facecolor='none')
                self.image['axes'].add_patch(area)
        except NameError:
            raise NameError('Image was not added yet. Run .add_image() first!')

        return True

    def add_histogram(self, image_from_spectra = False, plot_style = None, bins = 'auto'):
        # Set plotting style
        if plot_style is not None:
            plt.style.use(plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

        axes.set_ylabel('Frequency')
        if self.method == 'sum':
            axes.set_xlabel('PL-Intensity (counts/second)')
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

        axes.hist(self.histogram_data, bins = bins)

        # Add figure and axes for to self further manipulation
        self.histogram = {'figure': figure, 'axes': axes}

        return True


    def save_image(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.image['figure'].savefig(title, bbox_inches='tight', transparent=True)

        return True

    def save_histogram(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.histogram['figure'].savefig(title, bbox_inches='tight', transparent=True)

        return True


class DataHyperSpectral:
    def __init__(self, file_name, spectral_ranges, unit_spectral_range='eV', folder_name='.'):

        self.spectral_ranges = spectral_ranges
        self.unit_spectral_range = unit_spectral_range
        self.file_name = file_name
        self.folder_name = folder_name
        self.spectral_scans = {}
        for n, spectral_range in enumerate(self.spectral_ranges):
            self.spectral_scans[n] = DataConfocalScan(self.file_name, self.folder_name, spectral_range,
                                                      self.unit_spectral_range)

    def add_hyperspectral_image(self, masking_treshold, scale_bar=None, plot_style=None, interpolation=None):
        def get_mask(image_data, n):
            mask = {}
            for k in image_data:
                if k != n:
                    mask[k] = image_data[n] > image_data[k]

            sum_mask = np.zeros(image_data[n].shape)
            for k in mask:
                sum_mask += mask[k]

            final_mask = np.zeros(image_data[n].shape)
            for i in range(len(sum_mask)):
                for j in range(len(sum_mask[i])):
                    if sum_mask[i][j] < len(image_data) - 1:
                        final_mask[i][j] = 0
                    else:
                        final_mask[i][j] = 1

            return final_mask

        # Define function for formatting scale bar text
        def format_scale_bar(scale_value, unit='um'):
            if unit == 'um':
                scale_string = r'{0}'.format(int(scale_value)) + r' $\mathrm{\mu}$m'
            return scale_string

        # Set color maps
        color_maps = ['Reds_r', 'Greens_r', 'Blues_r', 'Purples_r']

        # Set plotting style
        if plot_style != None:
            plt.style.use(plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

        # Normalize image data
        self.image_data_to_plot = {}
        for n in range(len(self.spectral_ranges)):
            self.image_data_to_plot[n] = (self.spectral_scans[n].image_data_from_spectra - np.min(
                self.spectral_scans[n].image_data_from_spectra))
            self.image_data_to_plot[n] = self.image_data_to_plot[n] / np.max(self.image_data_to_plot[n])

        # Mask and plot image data
        for n in range(len(self.spectral_ranges)):
            if masking_treshold != None:
                mask = self.image_data_to_plot[n] <= masking_treshold
            else:
                mask = get_mask(self.image_data_to_plot, n)
            self.image_data_to_plot[n] = np.ma.masked_array(self.image_data_to_plot[n], mask)
            im = axes.imshow(self.image_data_to_plot[n],
                             cmap=plt.get_cmap(color_maps[n]),
                             extent=[self.spectral_scans[n].extent['x_min'], self.spectral_scans[n].extent['x_max'],
                                     self.spectral_scans[n].extent['y_min'], self.spectral_scans[n].extent['y_max']],
                             interpolation=interpolation)

        # Add scale bar
        if scale_bar is not None:
            default_scale_bar = {'scale': 5, 'font_size': 24, 'color': 'white', 'position': 'lower left'}
            if not isinstance(scale_bar, dict):
                default_scale_bar['scale'] = scale_bar
                scale_bar = default_scale_bar
            else:
                for key in default_scale_bar:
                    if key not in scale_bar:
                        scale_bar[key] = default_scale_bar[key]
            fontprops = fm.FontProperties(size=scale_bar['font_size'])
            scalebar = AnchoredSizeBar(axes.transData,
                                       scale_bar['scale'],
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
        plt.axis('off')

        # Set nan pixel to red
        color_map = mpl.cm.get_cmap()
        color_map.set_bad(color = 'white', alpha = 0)

        # Add figure and axes for to self further manipulation
        self.image = {'figure': figure, 'axes': axes}

        return True

    def save_image(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.image['figure'].savefig(title, bbox_inches='tight', transparent=True, facecolor='black')

        return True
