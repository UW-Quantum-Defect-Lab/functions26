# 2021-05-01
# This code was made for use in the Fu lab
# by Christian Zimmermann

import numpy as np
import scipy.io as spio
import warnings

from sif_reader import np_open as readsif
from .constants import conversion_factor_nm_to_ev  # eV*nm
from .constants import n_air
from .DataDictXXX import DataDictFilenameInfo
from .DataDictXXX import DataDictSIF
from .Plot2D import two_dimensional_plot




class Data2D:
    def __init__(self, file_name, folder_name, allowed_file_extensions):
        self.file_name = file_name
        if file_name == '':
            raise RuntimeError('File name is an empty string')
        self.folder_name = folder_name
        self.file_extension = self.file_name.split('.')[-1]
        self.check_file_type(allowed_file_extensions)

        self.file_info = DataDictFilenameInfo()
        self.get_file_info()

        self.data = np.zeros((1, 1))
        self.x_axis = [0]
        self.y_axis = [0]
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


class DataMagnetoPL(Data2D):
    allowed_file_extensions = ['mat']

    def __init__(self, file_name, folder_name='.', background = 0, wavelength_offset = 0,
                        x_axis_identifier = 'eV',
                        refractive_index = n_air, second_order = True):
        self.background = background
        self.wavelength_offset = wavelength_offset
        self.refractive_index = refractive_index
        self.second_order = second_order

        self.x_axis_identifier = x_axis_identifier

        super().__init__(file_name, folder_name, self.allowed_file_extensions)

    def get_data(self):
        matlab_file_data = spio.loadmat(self.file_name)

        self.wavelength = matlab_file_data['pled'][0][0][0][0] + self.wavelength_offset
        if self.second_order:
            self.photon_energy = conversion_factor_nm_to_ev/(self.wavelength/2*self.refractive_index)
        else:
            self.photon_energy = conversion_factor_nm_to_ev/(self.wavelength*self.refractive_index)
        if self.x_axis_identifier == 'nm':
            self.x_axis = self.wavelength
        elif self.x_axis_identifier == 'eV':
            self.x_axis = self.photon_energy

        self.magnetic_field = matlab_file_data['pled'][0][0][14][0]
        self.y_axis = self.magnetic_field

        self.data = matlab_file_data['pled'][0][0][29] - self.background
        self.data_normalized = []
        for spectrum in self.data:
            self.data_normalized.append(spectrum/np.nanmax(spectrum))
        self.data_normalized = np.array(self.data_normalized)

        return True

    def add_heatmap(self,
                    axes_limits = 'Auto',
                    scale = 'Auto',
                    color_bar = True,
                    shading = 'auto',
                    plot_style = None,
                    plot_normalized_spectra = False):
        if self.x_axis_identifier == 'eV':
            x_axis_label = 'Photon Energy (eV)'
        elif self.x_axis_identifier == 'nm':
            x_axis_label = 'Wavelength (nm)'
        y_axis_label = 'Magnetic Field (T)'
        if plot_normalized_spectra:
            color_bar_label = 'Normalized PL-Intensity (rel. units)'
        else:
            color_bar_label = 'PL-Intensity (counts/second)'

        if plot_normalized_spectra:
            data = self.data_normalized
        else:
            data = self.data

        self.heatmap = two_dimensional_plot(data, self.x_axis, self.y_axis,
                                            x_axis_label = x_axis_label,
                                            y_axis_label = y_axis_label,
                                            axes_limits = axes_limits,
                                            scale = scale,
                                            color_bar = color_bar,
                                            color_bar_label = color_bar_label,
                                            shading = shading,
                                            plot_style = plot_style)

        return True


class DataSIFKineticSeries(Data2D):
    allowed_file_extensions = ['sif']

    def __init__(self, file_name, folder_name='.', second_order = False, wavelength_offset = 0, background_per_cycle = 300, refractive_index = n_air):
        self.second_order = second_order
        self.refractive_index = refractive_index

        self.infoSIF = DataDictSIF()
        self.infoSIF['wavelength_offset_nm'] = wavelength_offset
        self.infoSIF['background_counts_per_cycle'] = background_per_cycle

        super().__init__(file_name, folder_name, self.allowed_file_extensions)

        self.set_background()
        self.set_all_x_data()
        self.set_y_data_nobg_counts_per_second()

    def get_file_info(self):

        # Save filename without folder and file extension
        file_info_raw = self.file_name.split('.')[-2]
        if '/' in self.file_name:
            file_info_raw = file_info_raw.split('/')[-1]

        file_info_raw_components = file_info_raw.split('_')  # All file info are separated by '_'
        self.file_info.get_info(file_info_raw_components)  # retrieve info from file

        return True

    def get_data(self):
        counts_info, acquisition_info = readsif(self.folder_name + '/' + self.file_name)
        self.data = []
        for n in range(len(counts_info)):
            self.data.append(list(counts_info[n][0]))
        self.data = np.array(self.data)
        self.data = np.flip(self.data, axis = 1)

        self.data_normalized = []
        for spectrum in self.data:
            self.data_normalized.append(spectrum/np.nanmax(spectrum))
        self.data_normalized = np.array(self.data_normalized)

        self.x_axis = {}
        x_pixel = np.array([pixel for pixel in range(1, len(self.data[0])+1, 1)])
        self.x_axis['x_pixel'] = x_pixel[::-1]

        self.infoSIF['cal_data'] = acquisition_info['Calibration_data']
        self.infoSIF['exposure_time_secs'] = acquisition_info['ExposureTime']
        self.infoSIF['cycles'] = acquisition_info['AccumulatedCycles']

        self.y_axis = np.array([n*acquisition_info['CycleTime'] for n in range(len(self.data))])

        return True

    def get_wavelength_calibration(self):
        return self.infoSIF['cal_data'][0] \
               + self.infoSIF['cal_data'][1] * self.x_axis['x_pixel'] \
               + self.infoSIF['cal_data'][2] * self.x_axis['x_pixel'] ** 2 \
               + self.infoSIF['cal_data'][3] * self.x_axis['x_pixel'] ** 3 \
               + self.infoSIF['wavelength_offset_nm']

    def set_x_data_in_nm(self):
        self.x_axis['x_nm'] = self.get_wavelength_calibration()
        return True

    def set_x_data_in_nm_2nd_order(self):
        self.x_axis['x_nm'] = self.get_wavelength_calibration()/2.
        return True

    def set_x_data_in_ev(self):
        self.x_axis['x_eV'] = conversion_factor_nm_to_ev/(self.get_wavelength_calibration()*self.refractive_index)
        return True

    def set_x_data_in_ev_2nd_order(self):
        self.x_axis['x_eV'] = conversion_factor_nm_to_ev/(self.get_wavelength_calibration()*self.refractive_index/2.)
        return True

    def set_all_x_data(self):
        if self.second_order:
            self.set_x_data_in_nm_2nd_order()
            self.set_x_data_in_ev_2nd_order()
        else:
            self.set_x_data_in_nm()
            self.set_x_data_in_ev()

    def set_background(self):
        self.infoSIF['background_counts_per_second'] = self.infoSIF['background_counts_per_cycle'] / \
                                                       self.infoSIF['exposure_time_secs']

        return True

    def set_y_data_nobg_counts_per_second(self):
        self.data = self.data/self.infoSIF['exposure_time_secs']/self.infoSIF['cycles']
        self.data = self.data - self.infoSIF['background_counts_per_second']

        return True

    def add_heatmap(self,
                    axes_limits = 'Auto',
                    scale = 'Auto',
                    color_bar = True,
                    shading = 'auto',
                    plot_style = None,
                    plot_normalized_spectra = False):
        x_axis_label = 'Photon Energy (eV)'
        y_axis_label = 'Time (s)'
        if plot_normalized_spectra:
            color_bar_label = 'Normalized PL-Intensity (rel. units)'
        else:
            color_bar_label = 'PL-Intensity (counts/second)'

        if plot_normalized_spectra:
            data = self.data_normalized
        else:
            data = self.data

        self.heatmap = two_dimensional_plot(data, self.x_axis['x_eV'], self.y_axis,
                                            x_axis_label = x_axis_label,
                                            y_axis_label = y_axis_label,
                                            axes_limits = axes_limits,
                                            scale = scale,
                                            color_bar = color_bar,
                                            color_bar_label = color_bar_label,
                                            shading = shading,
                                            plot_style = plot_style)

        return True
