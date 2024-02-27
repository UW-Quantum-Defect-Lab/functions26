# 2019-11-19 and last update on 2020-09-14
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# DataSIF modified by Christian Zimmermann


import csv
from typing import Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as spo
import scipy.io as spio
import warnings

from matplotlib import pyplot as plt
from sif_reader import np_open as readsif
from .constants import conversion_factor_nm_to_ev  # eV*nm
from .constants import n_air
from .DataDictXXX import DataDictSpectrum, DataDictSIF, DataDictOP, DataDictOP2LaserDelay, DataDictT1
from .DataDictXXX import DataDictFilenameInfo
from .DataFrame26 import DataFrame26
from .Dict26 import Dict26
from .useful_functions import get_added_label_from_unit
from .FittingManager import exp_with_bg_fit_turton_poison, double_exp_with_bg_fit_turton_poison, voigt_linear_fit, \
    FittingManager
from .units import unit_families


class DataXXX:

    def __init__(self, file_name, folder_name='.', default_keys=None, allowed_units=None, allowed_file_extensions=None,
                 spacer=' ', qdlf_datatype=None):
        self.file_name = file_name
        self.folder_name = folder_name
        self.default_keys = default_keys
        self.allowed_units = allowed_units
        self.allowed_file_extensions = allowed_file_extensions
        self.spacer = spacer
        self.qdlf_datatype = qdlf_datatype

        if len(self.file_name.split('/')) > 1:
            self.folder_name = '/'.join(self.file_name.split('/')[:-1])
            self.file_name = self.file_name.split('/')[-1]

        self.data = DataFrame26(default_keys, allowed_units, spacer, qdlf_datatype=self.qdlf_datatype)
        self.labels = Dict26(default_keys, allowed_units, spacer)

        self.file_extension = self.file_name.split('.')[-1]

        if file_name == '':
            raise ValueError('File name is an empty string')

        self.check_file_type(allowed_file_extensions)
        if self.file_extension == 'qdlf':
            from .filing.QDLFiling import QDLFDataManager
            qdlf_mng = QDLFDataManager.load(filename=self.folder_name + '/' + self.file_name)
            # check if this is a processed qdlf, or just a data file like the optical pumping qdlf files.
            if all([parameter in qdlf_mng.parameters for parameter in ['filename info', 'additional info', 'labels']]):
                self.data = qdlf_mng.data
                self.file_info = DataDictFilenameInfo(**qdlf_mng.parameters['filename info'])
                self.set_additional_info(qdlf_mng.parameters['additional info'])
                self.labels = Dict26(default_keys, allowed_units, self.spacer, **qdlf_mng.parameters['labels'])
                self.__post_load_qdlf__()
                return

        # is executed if 1. file extension is not qdlf and 2. if qdlf file does not contain filename info etc, meaning
        # it is not a processed file
        self.file_info = DataDictFilenameInfo()
        self.get_file_info()
        if self.get_data():
            self.__post_init__()

    def __post_init__(self):
        # Define your own if needed. Used for post-initialization processing of data
        pass

    def __post_load_qdlf__(self):
        # Define your own if needed. Used for post-initialization processing of qdlf data
        pass

    def get_data(self):
        warnings.warn('Define your own get_data() function')
        pass

    def get_additional_info(self):
        # Define your own of needed
        return dict()

    def set_additional_info(self, info_dictionary):
        # write your own if needed
        pass

    def get_file_info(self):

        # Save filename without folder and file extension
        file_info_raw = '.'.join(self.file_name.split('.')[:-1])
        if '/' in self.file_name:
            file_info_raw = file_info_raw.split('/')[-1]

        file_info_raw_components = file_info_raw.split('_')  # All file info are separated by '_'
        self.file_info.get_info(file_info_raw_components)  # retrieve info from file

        return True

    def check_file_type(self, allowed_file_extensions):
        allowed_file_extensions = [fe.lower() for fe in allowed_file_extensions]
        if self.file_extension.lower() not in allowed_file_extensions:
            raise ValueError('Given file extension does not much the allowed extensions: '
                             + str(allowed_file_extensions))

    # @classmethod
    # def load_with_qdlf_manager(cls, filename):
    #     filename = filename
    #     return cls(filename)

    def get_qdlf_manager(self):
        from .filing.QDLFiling import QDLFDataManager
        additional_info = self.get_additional_info()
        filename_info = self.file_info
        labels = self.labels
        all_info = {'additional info': dict(additional_info), 'filename info': dict(filename_info),
                    'labels': dict(labels)}

        return QDLFDataManager(data=self.data, parameters=all_info, datatype=self.qdlf_datatype)

    def save_with_qdlf_data_manager(self, filename=''):
        if filename == '':
            filename = self.folder_name + '/' + '.'.join(self.file_name.split('.')[:-1])
        qdlf_mng = self.get_qdlf_manager()
        qdlf_mng.save(filename)


class DataSpectrum(DataXXX):
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Time': unit_families['Time']}
    default_keys = ['x_pixel', 'x_nm', 'x_eV',
                    'y_counts', 'y_counts_per_cycle',
                    'y_counts_per_second', 'y_counts_per_second_per_power',
                    'y_counts_per_cycle',
                    'y_nobg_counts_per_second', 'y_nobg_counts_per_second_per_power']
    spacer = '_'

    def __init__(self, file_name, second_order=False, wavelength_offset=0, refractive_index=n_air,
                 background_per_cycle=300, folder_name='.', from_video=True):

        self.from_video = from_video
        self.second_order = second_order
        self.refractive_index = refractive_index

        self.infoSpectrum = DataDictSpectrum()
        self.infoSpectrum['wavelength_offset_nm'] = wavelength_offset
        self.infoSpectrum['background_counts_per_cycle'] = background_per_cycle

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer,
                         self.qdlf_datatype)

    def __post_init__(self):
        self.set_background()
        self.set_all_y_data()
        self.set_all_x_data()
        self.total_counts = self.integrate_counts()  # background is taken into account

    def get_additional_info(self):
        return {'infoSpectrum': dict(self.infoSpectrum), 'from_video': self.from_video,
                'second_order': self.second_order,
                'total_counts': self.total_counts}

    def set_additional_info(self, info_dictionary):
        self.infoSpectrum = DataDictSIF(**info_dictionary['infoSpectrum'])
        self.from_video = info_dictionary['from_video']
        self.second_order = info_dictionary['second_order']
        self.total_counts = info_dictionary['total_counts']

    def get_wavelength_calibration(self):
        return self.infoSpectrum['cal_data'][0] \
               + self.infoSpectrum['cal_data'][1] * self.data.x_pixel \
               + self.infoSpectrum['cal_data'][2] * self.data.x_pixel ** 2 \
               + self.infoSpectrum['cal_data'][3] * self.data.x_pixel ** 3 \
               + self.infoSpectrum['wavelength_offset_nm']

    def integrate_counts(self, unit='counts', subtract_bg=False):
        tot = self.data['y_{0}'.format(unit)].sum()
        if subtract_bg:
            tot -= len(self.data) * self.infoSpectrum['background_{0}'.format(unit)]
        return tot

    def integrate_in_region(self, start, end, unit_x='pixel', unit_y='counts', subtract_bg=False):
        data_in_region = self.data.loc[(self.data['x_{0}'.format(unit_x)] >= start)
                                       & (self.data['x_{0}'.format(unit_x)] <= end)]
        if subtract_bg:
            data_in_region -= self.infoSpectrum['background_{0}'.format(unit_y)]
        return data_in_region['y_{0}'.format(unit_y)].sum()

    def shift_counts(self, shift_y, unit='counts', new_end_string='_shifted'):
        self.data['y_{0}{1}'.format(unit, new_end_string)] = self.data['y_{0}'.format(unit)] + shift_y
        return True

    def renormalize_counts(self, times_renorm_value, unit='counts',
                           added_string_to_unit='_renormalized', added_string_to_label=None):

        if added_string_to_label is None:
            added_string_to_label = get_added_label_from_unit(added_string_to_unit)

        self.data['y_{0}'.format(unit + added_string_to_unit)] = self.data['y_{0}'.format(unit)] * times_renorm_value
        self.labels['y_{0}'.format(unit + added_string_to_unit)] = self.labels['y_{0}'.format(unit)] \
                                                                   + added_string_to_label
        return True

    def renormalize_counts_to_max(self, unit='counts', added_string_to_unit='_renormalized_to_max',
                                  added_string_to_label='Renormalized to max'):
        factor = 1. / self.data['y_{0}'.format(unit)].max()
        self.renormalize_counts(factor, unit, added_string_to_unit, added_string_to_label)

        return True

    def set_x_data_in_nm(self):
        self.data['x_nm'] = self.get_wavelength_calibration()
        self.labels['x_nm'] = 'Wavelength (nm)'
        return True

    def set_x_data_in_nm_2nd_order(self):
        self.data['x_nm'] = self.get_wavelength_calibration() / 2.
        self.labels['x_nm'] = 'Wavelength (nm)'
        return True

    def set_x_data_in_ev(self):
        self.data['x_eV'] = conversion_factor_nm_to_ev/(self.get_wavelength_calibration()*self.refractive_index)
        self.labels['x_eV'] = 'Photon Energy (eV)'
        return True

    def set_x_data_in_ev_2nd_order(self):
        self.data['x_eV'] = conversion_factor_nm_to_ev/(self.get_wavelength_calibration()*self.refractive_index/2.)
        self.labels['x_eV'] = 'Photon Energy (eV)'
        return True

    def set_all_x_data(self):
        if self.second_order:
            self.set_x_data_in_nm_2nd_order()
            self.set_x_data_in_ev_2nd_order()
        else:
            self.set_x_data_in_nm()
            self.set_x_data_in_ev()

    def set_background(self):
        self.infoSpectrum['background_counts_per_second'] = self.infoSpectrum['background_counts_per_cycle'] / \
                                                            self.infoSpectrum['exposure_time_secs']

        self.infoSpectrum['background_counts'] = self.infoSpectrum['background_counts_per_cycle'] * \
                                                 self.infoSpectrum['cycles']

        power = self.file_info['Lsr: Power (nW)']
        try:
            self.infoSpectrum['background_counts_per_second_per_power'] = self.infoSpectrum['background_counts'
                                                                                            '_per_second'] / power
        except (ValueError, TypeError, ZeroDivisionError):
            warnings.warn('No power information found in file_info. Y data per power were not calculated.')

        return True

    def set_y_data_counts_per_second(self):
        self.data['y_counts_per_second'] = self.data.y_counts / self.infoSpectrum['exposure_time_secs'] / \
                                           self.infoSpectrum['cycles']

        self.labels['y_counts_per_second'] = 'Counts/sec'
        return True

    def set_y_data_counts_per_cycle(self):
        self.data['y_counts_per_cycle'] = self.data.y_counts / self.infoSpectrum['cycles']

        self.labels['y_counts_per_cycle'] = 'Counts/cycle'
        return True

    def set_y_data_counts_per_second_per_power(self):
        power = self.file_info['Lsr: Power (nW)']
        try:
            self.data['y_counts_per_second_per_power'] = self.data['y_counts_per_second'] / power
            return True
        except (ValueError, TypeError):
            return False

    def set_y_data_nobg_counts_per_second(self):
        self.data['y_nobg_counts_per_second'] = self.data['y_counts_per_second'] \
                                                - self.infoSpectrum['background_counts_per_second']

        self.labels['y_nobg_counts_per_second'] = 'Counts/sec'
        return True

    def set_y_data_nobg_counts_per_cycle(self):
        self.data['y_nobg_counts_per_cycle'] = self.data['y_counts_per_cycle'] \
                                               - self.infoSpectrum['background_counts_per_cycle']

        self.labels['y_nobg_counts_per_cycle'] = 'Counts/cycle'
        return True

    def set_y_data_nobg_counts_per_second_per_power(self):
        try:
            self.data['y_nobg_counts_per_second_per_power'] = self.data['y_counts_per_second_per_power'] \
                                                              - self.infoSpectrum['background_counts_per'
                                                                                  '_second_per_power']
            return True
        except KeyError:
            return False

    def set_all_y_data(self):
        self.set_y_data_counts_per_second()
        self.set_y_data_counts_per_cycle()
        self.set_y_data_counts_per_second_per_power()

        self.set_y_data_nobg_counts_per_second()
        self.set_y_data_nobg_counts_per_cycle()
        self.set_y_data_nobg_counts_per_second_per_power()

        return True

    def get_wavelength_offset(self, fit_output_points: Union[None, int] = None,
                              want_plot: bool = False, fig: plt.Figure = None, ax: plt.Axes = None,
                              data_plot_options: Dict = None, fit_plot_options: Dict = None) \
            -> Tuple[float, List[float], List[float], FittingManager]:
        """
        This function is used to find the offset between the spectrometer wavelength and the input wavelength of a
        directly reflected beam. It purposefully does not return only the offset, so that the user will always be
        reminded to check the fit and not wholeheartedly trust this highly automated fitting method.

        Parameters
        ----------
        fit_output_points: int
            An integer indicating the length of the x_fit and y_fit plotted and returned lists.
        want_plot: bool
            If True, plots the data and the attempted fit
        fig: plt.Figures
            If an ax is not given, use the figure to get its current axes to plot the data and the fit.
            Defaults to None. If None, plots on the current (or new) figure of matplotlib.pyplot.
        ax: plt.Axes
            Axes to plot in. Has the highest priority. Defaults to None.
        data_plot_options: Dict
            A dictionary of all the desired plotting options for the data as used in matplotlib.pyplot.plot function.
            Defaults to None.
        fit_plot_options: Dict
            A dictionary of all the desired plotting options for the data as used in matplotlib.pyplot.plot function.
            Defaults to None.

        Returns
        -------
        float, List[float], List[float], FittingManager
            Returns offset, x_fit_data, y_fit_data, fitting_manager: FittingManager. The offset (strictly in first order
            and nanometers) can be readily plugged in another DataSIF function. The fitted data-points are given in case
            the use wants them for an arbitrary application. The FittingManager is given if more information about the
            fit are needed. Even though the fitting manager contains the fitted data hence making the return of
            x_fit_data, and y_fit_data redundant, I think it will be easier for the user to have direct access to them.
        """

        if data_plot_options is None:
            data_plot_options = {}
        if fit_plot_options is None:
            fit_plot_options = {}

        x_data = self.data['x_nm']
        if self.second_order:
            x_data *= 2
        y_data = self.data['y_nobg_counts_per_second']

        real_wavelength = self.file_info['Lsr: Wavelength (nm)']
        fitmng, _, fit_center = voigt_linear_fit(x_data, y_data)

        if fit_output_points is None:
            fit_output_points = len(y_data) * 10
        fitmng.get_x_y_fit(x_min=None, x_max=None, output_points=fit_output_points)

        offset = real_wavelength - fit_center

        if want_plot:
            if ax is not None:
                ax.plot(x_data, y_data, **data_plot_options)
                ax.plot(fitmng.x_fit, fitmng.y_fit, **fit_plot_options)
            elif fig is not None:
                ax = fig.gca()
                ax.plot(x_data, y_data, **data_plot_options)
                ax.plot(fitmng.x_fit, fitmng.y_fit, **fit_plot_options)
            else:
                plt.plot(x_data, y_data, **data_plot_options)
                plt.plot(fitmng.x_fit, fitmng.y_fit, **fit_plot_options)

        return offset, fitmng.x_fit, fitmng.y_fit, fitmng


class DataSIF(DataSpectrum):
    allowed_file_extensions = ['sif', 'qdlf']
    qdlf_datatype = 'sif'

    def __init__(self, file_name, second_order=False, wavelength_offset=0, refractive_index=n_air,
                 background_per_cycle=300, folder_name='.', from_video=True):

        super().__init__(file_name, second_order, wavelength_offset, refractive_index, background_per_cycle,
                         folder_name, from_video)

        # used in case people have used it in the past
        self.infoSIF = self.infoSpectrum

    def get_data(self):
        counts_info, acquisition_info = readsif(self.folder_name + '/' + self.file_name)
        counts_extended_list = counts_info.tolist()
        y_counts = counts_extended_list[0][0]
        self.data['y_counts'] = y_counts

        x_pixel = np.array([pixel for pixel in range(1, len(y_counts) + 1, 1)])
        self.data['x_pixel'] = x_pixel

        self.infoSpectrum['cal_data'] = acquisition_info['Calibration_data']
        self.infoSpectrum['exposure_time_secs'] = acquisition_info['ExposureTime']
        if not self.from_video:
            self.infoSpectrum['cycles'] = acquisition_info['AccumulatedCycles']
        else:
            self.infoSpectrum['cycles'] = 1

        self.labels['x_pixel'] = 'Pixel'
        self.labels['y_counts'] = 'Counts'

        return True


class DataOP(DataXXX):
    # https://core.ac.uk/download/pdf/82106638.pdf to add fitting functions with irregular binning.
    allowed_file_extensions = ['csv', 'qdlf', 'json']
    allowed_units = {'Time': unit_families['Time']}
    default_keys = ['x_time_us', 'y_counts', 'y_counts_per_cycle', 'y_normalized_counts']
    spacer = '_'
    qdlf_datatype = 'op'

    def __init__(self, file_name, folder_name='.', run_simple_fit_functions=False, run_turton_poison_fit_functions=True,
                 turton_poison_fit_decay_option='single', fit_start_end_indices=(None, None)):

        self.infoOP = DataDictOP()
        self.size = -1
        self.simple_fit = None
        self.turton_poison_fit = None
        self.time_step = None
        self.run_simple_fit_functions = run_simple_fit_functions
        self.run_turton_poison_fit_functions = run_turton_poison_fit_functions
        self.turton_poison_fit_decay_option = turton_poison_fit_decay_option
        self.fit_start_end_indices = fit_start_end_indices

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer,
                         self.qdlf_datatype)

    def __post_init__(self):
        self.y_data_in_normalized_on_peak_counts()
        if self.infoOP['numRun'] is not None:
            self.y_data_in_counts_per_cycle()
            self.y_data_in_counts_per_cycle_per_time_step()
        if self.run_simple_fit_functions:
            try:
                self.get_simple_fit(units_y='counts')
                self.get_simple_fit(units_y='normalized_counts')
                if self.infoOP['numRun'] is not None:
                    self.get_simple_fit(units_y='counts_per_cycle')
                    self.get_simple_fit(units_y='counts_per_cycle_per_time_step')
            except (ValueError, RuntimeError) as e:
                warnings.warn('Fit was not possible: ' + e)

        if self.run_turton_poison_fit_functions:
            try:
                self.get_turton_poison_fit(units_y='counts')
                self.get_turton_poison_fit(units_y='normalized_counts')
                if self.infoOP['numRun'] is not None:
                    self.get_turton_poison_fit(units_y='counts_per_cycle')
                    self.get_turton_poison_fit(units_y='counts_per_cycle_per_time_step')
            except (ValueError, RuntimeError) as e:
                warnings.warn('Turton-Poison fit was not possible: ' + e)

    def get_data(self):

        self.labels['x_time_us'] = 'Time (us)'
        self.labels['y_counts'] = 'Counts (Arb. Units)'
        file_name = self.folder_name + '/' + self.file_name
        from .filing.QDLFiling import QDLFDataManager
        file = QDLFDataManager.load(file_name)
        if file.datatype == 'op':
            self.size = len(file.data['x1'])
            self.time_step = file.data['x1'][1] - file.data['x1'][0]

            self.data['x_time_us'] = file.data['x1']
            self.data['y_counts'] = file.data['y1']

            # getting rest info
            self.infoOP['numRun'] = file.parameters['Measurement Cycle Number']
            self.infoOP['numPerRun'] = 1

            self.infoOP['pumpOnTime_us'] = file.parameters['AOM/Pump on time'].us
            self.infoOP['pumpOffTime_us'] = file.parameters['AOM/Pump off time'].us
            return True
        elif file.datatype is None:
            # This is the old method we collected data (through Labview)
            self.size = len(file.data[0]) - 2
            if self.size < 1:
                self.size = -2  # means there is no data, only run values
                return False
            self.time_step = file.data[0][1] - file.data[0][0]

            self.data['x_time_us'] = file.data[0][:-2]
            self.data['y_counts'] = file.data[1][:-2]

            # getting rest info
            self.infoOP['numRun'] = file.data[0][-2]
            self.infoOP['numPerRun'] = file.data[1][-2]

            self.infoOP['pumpOnTime_us'] = file.data[0][-1]
            self.infoOP['pumpOffTime_us'] = file.data[1][-1]

            if file.data[0][-1] - file.data[0][-2] != \
                    file.data[0][-2] - file.data[0][-3]:
                warnings.warn('This file might be a T1 file')
            return True
        else:
            return False

    def get_additional_info(self):
        return {'infoOP': dict(self.infoOP), 'size': self.size, 'simple_fit': self.simple_fit,
                'time_step': self.time_step, 'run_simple_fit_functions': self.run_simple_fit_functions,
                'turton_poison_fit': self.turton_poison_fit,
                'run_turton_poison_fit_functions': self.run_turton_poison_fit_functions,
                'turton_poison_fit_decay_option': self.turton_poison_fit_decay_option}

    def set_additional_info(self, info_dictionary):
        self.infoOP = DataDictOP(**info_dictionary['infoOP'])
        self.size = info_dictionary['size']
        self.simple_fit = info_dictionary['simple_fit']
        self.time_step = info_dictionary['time_step']
        self.run_simple_fit_functions = info_dictionary['run_simple_fit_functions']
        self.turton_poison_fit = info_dictionary['turton_poison_fit']
        self.run_turton_poison_fit_functions = info_dictionary['run_turton_poison_fit_functions']
        self.turton_poison_fit_decay_option = info_dictionary['turton_poison_fit_decay_option']

    def y_data_in_counts_per_cycle(self):
        rescaling_factor_cycle = (self.infoOP['numRun'] * self.infoOP['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle'] = [counts / rescaling_factor_cycle for counts in self.data.y_counts]
        self.labels['y_counts_per_cycle'] = 'Counts/cycle'
        return True

    def y_data_in_normalized_on_peak_counts(self):
        rescaling_factor_normalize = self.data.y_counts.max()  # normalized
        self.data['y_normalized_counts'] = [counts / rescaling_factor_normalize for counts in self.data.y_counts]
        self.labels['y_normalized_counts'] = 'Counts normalized to max'
        return True

    def y_data_in_counts_per_cycle_per_time_step(self):
        rescaling_factor_cycle = (self.infoOP['numRun'] * self.infoOP['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle_per_time_step'] = [counts / rescaling_factor_cycle / self.time_step for counts in
                                                         self.data.y_counts]
        self.labels['y_counts_per_cycle_per_time_step'] = 'Counts/cycle/time_step'
        return True

    def get_signal_start_end_indices(self):
        start = int(self.size / 3) + 1
        end = 2 * int(self.size / 3) + 2
        return start, end

    def integrate_in_region(self, start, end, unit_x='time_us', unit_y='counts'):
        data_in_region = self.data.loc[(self.data['x_{0}'.format(unit_x)] >= start)
                                       & (self.data['x_{0}'.format(unit_x)] <= end)]
        return data_in_region['y_{0}'.format(unit_y)].sum()

    def average_in_region(self, start, end, unit_x='time_us', unit_y='counts'):
        data_in_region = self.data.loc[(self.data['x_{0}'.format(unit_x)] >= start)
                                       & (self.data['x_{0}'.format(unit_x)] <= end)]
        return data_in_region['y_{0}'.format(unit_y)].mean()

    def get_index_range_in_time_region(self, start, end, unit_x='time_us', include_end_point=False):
        if include_end_point:
            return (self.data['x_{0}'.format(unit_x)] >= start) & (self.data['x_{0}'.format(unit_x)] <= end)
        else:
            return (self.data['x_{0}'.format(unit_x)] >= start) & (self.data['x_{0}'.format(unit_x)] < end)

    def get_single_index_in_time(self, value, unit_x='time_us'):
        return (self.data['x_{0}'.format(unit_x)] >= value - self.time_step / 2) & (self.data['x_{0}'.format(unit_x)] <=
                                                                                    value + self.time_step)

    def get_starting_point(self):
        start = self.infoOP['pumpOnTime_us']
        x0 = float(self.data['x_time_us'][self.get_single_index_in_time(start)])
        return x0

    @staticmethod
    def simple_fit_function(x, bg, ampl, b, x0):
        return ampl * np.exp(-b * (x - x0)) + bg

    def get_simple_fit(self, units_y='counts'):
        from scipy.optimize import curve_fit
        units_y = 'y_{0}'.format(units_y)

        def estimate_p0(x, y, initial_counts_last_index=3, steady_state_first_index=None,
                        decay_estimation_final_index=None):
            if steady_state_first_index is None:
                steady_state_first_index = int(5 * len(y) / 6)
            if decay_estimation_final_index is None:
                decay_estimation_final_index = int(len(y) / 6)

            init_counts = np.mean(y[:initial_counts_last_index])
            steady_state_counts = np.mean(y[steady_state_first_index:])
            amp = init_counts - steady_state_counts
            bg = steady_state_counts
            x0 = list(x)[0]
            if amp < 0:
                return [bg, 0, 0, x0]

            reduced_y = y[1:decay_estimation_final_index] - bg
            # ignoring the negatives
            reduced_y = np.where(reduced_y <= 0, np.nan, reduced_y)

            b_estimations = -(np.log(reduced_y) - np.log(amp)) / (x[1:decay_estimation_final_index] - x0)

            # ignore nans
            b_estimations = np.ma.array(b_estimations, mask=np.isnan(b_estimations))
            b = np.mean(b_estimations)
            return [bg, amp, b, x0]

        start = self.infoOP['pumpOnTime_us'] if self.fit_start_end_indices[0] is None \
            else self.fit_start_end_indices[0]
        end = 2 * self.infoOP['pumpOnTime_us'] if self.fit_start_end_indices[1] is None \
            else self.fit_start_end_indices[1]
        x = self.data['x_time_us'][self.get_index_range_in_time_region(start, end)]
        y = self.data[units_y][self.get_index_range_in_time_region(start, end)]

        # we can only do the fitting if there are enough data points
        if len(y) < 5:
            return None

        estimated_p0 = estimate_p0(x, y)
        x0 = float(list(x)[0])

        if estimated_p0[1] > 0:
            popt, pcov = curve_fit(lambda x, bg, ampl, decay: self.simple_fit_function(x, bg, ampl, decay, x0),
                                   xdata=np.array(x, dtype=np.float64), ydata=np.array(y, dtype=np.float64),
                                   p0=estimated_p0[:-1])

            popt = [*popt, x0]
            pcov = np.vstack((np.array(pcov), np.array([0, 0, 0])))
            pcov = np.hstack((pcov, np.array([[0], [0], [0], [0]])))
            perr = np.sqrt(np.diag(pcov))
        else:
            popt, pcov = curve_fit(lambda x, bg: self.simple_fit_function(x, bg, 0, 0, x0),
                                   xdata=np.array(x, dtype=np.float64), ydata=np.array(y, dtype=np.float64),
                                   p0=estimated_p0[1])

            popt = [popt[0], 0, 0, x0]
            pcov_value = pcov[0][0]
            pcov = np.zeros(shape=(4, 4))
            pcov[0][0] = pcov_value
            perr = np.sqrt(np.diag(pcov))

        simple_fit = {'popt': popt, 'pcov': pcov, 'perr': perr, 'function': 'simple_fit_function'}

        if self.simple_fit is None:
            self.simple_fit = dict()
        self.simple_fit[units_y] = simple_fit

        return simple_fit

    def get_turton_poison_fit(self, units_y='counts'):
        units_y = 'y_{0}'.format(units_y)

        start = self.infoOP['pumpOnTime_us'] if self.fit_start_end_indices[0] is None \
            else self.fit_start_end_indices[0]
        end = 2 * self.infoOP['pumpOnTime_us'] if self.fit_start_end_indices[1] is None \
            else self.fit_start_end_indices[1]
        x = np.array(self.data['x_time_us'][self.get_index_range_in_time_region(start, end)])
        y = np.array(self.data[units_y][self.get_index_range_in_time_region(start, end)])

        # we can only do the fitting if there are enough data points
        if len(y) < 4:
            return None

        x0 = float(list(x)[0])

        if self.turton_poison_fit_decay_option == 'single':
            fitmng = exp_with_bg_fit_turton_poison(x - x0, y)
        elif self.turton_poison_fit_decay_option == 'double':
            fitmng = double_exp_with_bg_fit_turton_poison(x - x0, y)

        if self.turton_poison_fit is None:
            self.turton_poison_fit = dict()
        self.turton_poison_fit[units_y] = fitmng

        return fitmng

    def get_binned_data(self, step: int = 2):
        df = self.data
        new_df = DataFrame26(df.default_keys, df.allowed_units, df.spacer)

        for key in df.keys():
            data_list = np.array(df[key])
            if 'time' not in key:
                new_data_list = [sum(data_list[i:i + step]) for i in range(0, len(data_list), step)]
            else:
                new_data_list = [data_list[i] for i in range(0, len(data_list), step)]
            new_df[key] = new_data_list

        return new_df


class DataOP2LaserDelay(DataXXX):
    allowed_file_extensions = ['csv', 'qdlf', 'json']
    allowed_units = {'Time': unit_families['Time']}
    default_keys = ['x_time_us', 'y_counts', 'y_counts_per_cycle', 'y_normalized_counts']
    spacer = '_'
    qdlf_datatype = 'op'

    def __init__(self, file_name, folder_name='.', run_simple_fit_functions=False, run_turton_poison_fit_functions=True,
                 turton_poison_fit_decay_option='single', fit_start_end_indices=(None, None)):

        self.infoOP2LaserDelay = DataDictOP2LaserDelay()
        self.size = -1
        self.simple_fit = None
        self.turton_poison_fit = None
        self.time_step = None
        self.run_simple_fit_functions = run_simple_fit_functions
        self.run_turton_poison_fit_functions = run_turton_poison_fit_functions
        self.turton_poison_fit_decay_option = turton_poison_fit_decay_option
        self.fit_start_end_indices = fit_start_end_indices

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer,
                         self.qdlf_datatype)

    def __post_init__(self):
        self.y_data_in_normalized_on_peak_counts()
        if self.infoOP2LaserDelay['numRun'] is not None:
            self.y_data_in_counts_per_cycle()
            self.y_data_in_counts_per_cycle_per_time_step()

        self.set_control_data()
        self.set_signal_data()
        if self.run_simple_fit_functions:
            try:
                self.get_simple_fit(units_y='control_counts')
                self.get_simple_fit(units_y='control_normalized_counts')

                self.get_simple_fit(units_y='signal_counts')
                self.get_simple_fit(units_y='signal_normalized_counts')

                if self.infoOP2LaserDelay['numRun'] is not None:
                    self.get_simple_fit(units_y='control_counts_per_cycle')
                    self.get_simple_fit(units_y='control_counts_per_cycle_per_time_step')

                    self.get_simple_fit(units_y='signal_counts_per_cycle')
                    self.get_simple_fit(units_y='signal_counts_per_cycle_per_time_step')

            except (ValueError, RuntimeError) as e:
                warnings.warn('Fit was not possible: ' + e)

        if self.run_turton_poison_fit_functions:
            try:
                self.get_turton_poison_fit(units_y='control_counts')
                self.get_turton_poison_fit(units_y='control_normalized_counts')

                self.get_turton_poison_fit(units_y='signal_counts')
                self.get_turton_poison_fit(units_y='signal_normalized_counts')

                if self.infoOP2LaserDelay['numRun'] is not None:
                    self.get_turton_poison_fit(units_y='control_counts_per_cycle')
                    self.get_turton_poison_fit(units_y='control_counts_per_cycle_per_time_step')

                    self.get_turton_poison_fit(units_y='signal_counts_per_cycle')
                    self.get_turton_poison_fit(units_y='signal_counts_per_cycle_per_time_step')

            except (ValueError, RuntimeError) as e:
                warnings.warn('Turton-Poison fit was not possible: ' + e)

    def get_data(self):

        self.labels['x_time_us'] = 'Time (us)'
        self.labels['y_counts'] = 'Counts (Arb. Units)'
        file_name = self.folder_name + '/' + self.file_name
        from .filing.QDLFiling import QDLFDataManager
        file = QDLFDataManager.load(file_name)
        if file.datatype == 'op2laserdelay':
            self.size = len(file.data['x1'])
            self.time_step = file.data['x1'][1] - file.data['x1'][0]

            self.data['x_time_us'] = file.data['x1']
            self.data['y_counts'] = file.data['y1']

            # getting rest info
            self.infoOP2LaserDelay['numRun'] = file.parameters['Measurement Cycle Number']
            self.infoOP2LaserDelay['numPerRun'] = 1

            self.infoOP2LaserDelay['controlPumpOnTime_us'] = file.parameters['AOM/Control Pump on time'].us
            self.infoOP2LaserDelay['signalPumpOnTime_us'] = file.parameters['AOM/Signal Pump on time'].us
            self.infoOP2LaserDelay['controlSignalDelayTime_us'] = file.parameters['AOM/Control-Signal delay time'].us
            self.infoOP2LaserDelay['pumpOffTime_us'] = file.parameters['AOM/Pump off time'].us

            return True
        else:
            return False

    def get_additional_info(self):
        return {'infoOP2LaserDelay': dict(self.infoOP2LaserDelay), 'size': self.size, 'simple_fit': self.simple_fit,
                'time_step': self.time_step, 'run_simple_fit_functions': self.run_simple_fit_functions,
                'turton_poison_fit': self.turton_poison_fit,
                'run_turton_poison_fit_functions': self.run_turton_poison_fit_functions,
                'turton_poison_fit_decay_option': self.turton_poison_fit_decay_option}

    def set_additional_info(self, info_dictionary):
        self.infoOP = DataDictOP2LaserDelay(**info_dictionary['infoOP2LaserDelay'])
        self.size = info_dictionary['size']
        self.simple_fit = info_dictionary['simple_fit']
        self.time_step = info_dictionary['time_step']
        self.run_simple_fit_functions = info_dictionary['run_simple_fit_functions']
        self.turton_poison_fit = info_dictionary['turton_poison_fit']
        self.run_turton_poison_fit_functions = info_dictionary['run_turton_poison_fit_functions']
        self.turton_poison_fit_decay_option = info_dictionary['turton_poison_fit_decay_option']

    def y_data_in_counts_per_cycle(self):
        rescaling_factor_cycle = (self.infoOP2LaserDelay['numRun'] * self.infoOP2LaserDelay['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle'] = [counts / rescaling_factor_cycle for counts in self.data.y_counts]
        self.labels['y_counts_per_cycle'] = 'Counts/cycle'
        return True

    def y_data_in_normalized_on_peak_counts(self):
        rescaling_factor_normalize = self.data.y_counts.max()  # normalized
        self.data['y_normalized_counts'] = [counts / rescaling_factor_normalize for counts in self.data.y_counts]
        self.labels['y_normalized_counts'] = 'Counts normalized to max'
        return True

    def y_data_in_counts_per_cycle_per_time_step(self):
        rescaling_factor_cycle = (self.infoOP2LaserDelay['numRun'] * self.infoOP2LaserDelay['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle_per_time_step'] = [counts / rescaling_factor_cycle / self.time_step for counts in
                                                         self.data.y_counts]
        self.labels['y_counts_per_cycle_per_time_step'] = 'Counts/cycle/time_step'
        return True

    def get_pump_off_length_on_data_edges(self):
        total_measurement_length = int((self.infoOP2LaserDelay['controlPumpOnTime_us'] +
                                        self.infoOP2LaserDelay['signalPumpOnTime_us'] +
                                        self.infoOP2LaserDelay['controlSignalDelayTime_us']) / self.time_step)
        return int((self.size - total_measurement_length) / 2)

    def set_control_data(self):
        start, end = self.get_control_start_end_indices()
        self.data['x_control_time_us'] = self.data['x_time_us'][start:end]
        self.data['y_control_counts'] = self.data['y_counts'][start:end]
        self.data['y_control_normalized_counts'] = self.data['y_normalized_counts'][start:end]
        if self.infoOP2LaserDelay['numRun'] is not None:
            self.data['y_control_counts_per_cycle'] = self.data['y_counts_per_cycle'][start:end]
            self.data['y_control_counts_per_cycle_per_time_step'] = self.data['y_counts_per_cycle_per_time_step'][start:
                                                                                                                  end]

    def set_signal_data(self):
        start, end = self.get_signal_start_end_indices()
        self.data['x_signal_time_us'] = self.data['x_time_us'][start:end]
        self.data['y_signal_counts'] = self.data['y_counts'][start:end]
        self.data['y_signal_normalized_counts'] = self.data['y_normalized_counts'][start:end]
        if self.infoOP2LaserDelay['numRun'] is not None:
            self.data['y_signal_counts_per_cycle'] = self.data['y_counts_per_cycle'][start:end]
            self.data['y_signal_counts_per_cycle_per_time_step'] = self.data['y_counts_per_cycle_per_time_step'][start:
                                                                                                                 end]

    def get_control_start_end_indices(self):
        start = self.get_pump_off_length_on_data_edges()
        end = start + int(self.infoOP2LaserDelay['controlPumpOnTime_us'] / self.time_step) + 1
        return start, end

    def get_signal_start_end_indices(self):
        end = self.size - self.get_pump_off_length_on_data_edges() + 1
        start = end - int(self.infoOP2LaserDelay['signalPumpOnTime_us'] / self.time_step) - 1
        return start, end

    def integrate_in_region(self, start, end, unit_x='time_us', unit_y='counts'):
        data_in_region = self.data.loc[(self.data['x_{0}'.format(unit_x)] >= start)
                                       & (self.data['x_{0}'.format(unit_x)] <= end)]
        return data_in_region['y_{0}'.format(unit_y)].sum()

    def average_in_region(self, start, end, unit_x='time_us', unit_y='counts'):
        data_in_region = self.data.loc[(self.data['x_{0}'.format(unit_x)] >= start)
                                       & (self.data['x_{0}'.format(unit_x)] <= end)]
        return data_in_region['y_{0}'.format(unit_y)].mean()

    def get_index_range_in_time_region(self, start, end, unit_x='time_us', include_end_point=False):
        if include_end_point:
            return (self.data['x_{0}'.format(unit_x)] >= start) & (self.data['x_{0}'.format(unit_x)] <= end)
        else:
            return (self.data['x_{0}'.format(unit_x)] >= start) & (self.data['x_{0}'.format(unit_x)] < end)

    def get_single_index_in_time(self, value, unit_x='time_us'):
        return (self.data['x_{0}'.format(unit_x)] >= value - self.time_step / 2) & (self.data['x_{0}'.format(unit_x)] <=
                                                                                    value + self.time_step)

    @staticmethod
    def simple_fit_function(x, bg, ampl, b, x0):
        return ampl * np.exp(-b * (x - x0)) + bg

    def get_simple_fit(self, units_y):
        from scipy.optimize import curve_fit
        units_y = 'y_{0}'.format(units_y)

        def estimate_p0(x, y, initial_counts_last_index=3, steady_state_first_index=None,
                        decay_estimation_final_index=None):
            if steady_state_first_index is None:
                steady_state_first_index = int(5 * len(y) / 6)
            if decay_estimation_final_index is None:
                decay_estimation_final_index = int(len(y) / 6)

            init_counts = np.mean(y[:initial_counts_last_index])
            steady_state_counts = np.mean(y[steady_state_first_index:])
            amp = init_counts - steady_state_counts
            bg = steady_state_counts
            x0 = list(x)[0]
            if amp < 0:
                return [bg, 0, 0, x0]

            reduced_y = y[1:decay_estimation_final_index] - bg
            # ignoring the negatives
            reduced_y = np.where(reduced_y <= 0, np.nan, reduced_y)

            b_estimations = -(np.log(reduced_y) - np.log(amp)) / (x[1:decay_estimation_final_index] - x0)

            # ignore nans
            b_estimations = np.ma.array(b_estimations, mask=np.isnan(b_estimations))
            b = np.mean(b_estimations)
            return [bg, amp, b, x0]

        if units_y.startswith('y_control'):
            start, end = self.get_control_start_end_indices()
            start = start if self.fit_start_end_indices[0] is None else self.fit_start_end_indices[0]
            end = end if self.fit_start_end_indices[1] is None else self.fit_start_end_indices[1]
            x = np.array(self.data['x_control_time_us'][start:end])
        elif units_y.startswith('y_signal'):
            start, end = self.get_signal_start_end_indices()
            start = start if self.fit_start_end_indices[0] is None else self.fit_start_end_indices[0]
            end = end if self.fit_start_end_indices[1] is None else self.fit_start_end_indices[1]
            x = np.array(self.data['x_signal_time_us'][start:end])
        else:
            return None
        y = np.array(self.data[units_y][start:end])

        # we can only do the fitting if there are enough data points
        if len(y) < 5:
            return None

        estimated_p0 = estimate_p0(x, y)
        x0 = float(list(x)[0])

        if estimated_p0[1] > 0:
            popt, pcov = curve_fit(lambda x, bg, ampl, decay: self.simple_fit_function(x, bg, ampl, decay, x0),
                                   xdata=np.array(x, dtype=np.float64), ydata=np.array(y, dtype=np.float64),
                                   p0=estimated_p0[:-1])

            popt = [*popt, x0]
            pcov = np.vstack((np.array(pcov), np.array([0, 0, 0])))
            pcov = np.hstack((pcov, np.array([[0], [0], [0], [0]])))
            perr = np.sqrt(np.diag(pcov))
        else:
            popt, pcov = curve_fit(lambda x, bg: self.simple_fit_function(x, bg, 0, 0, x0),
                                   xdata=np.array(x, dtype=np.float64), ydata=np.array(y, dtype=np.float64),
                                   p0=estimated_p0[1])

            popt = [popt[0], 0, 0, x0]
            pcov_value = pcov[0][0]
            pcov = np.zeros(shape=(4, 4))
            pcov[0][0] = pcov_value
            perr = np.sqrt(np.diag(pcov))

        simple_fit = {'popt': popt, 'pcov': pcov, 'perr': perr, 'function': 'simple_fit_function'}

        if self.simple_fit is None:
            self.simple_fit = dict()
        self.simple_fit[units_y] = simple_fit

        return simple_fit

    def get_turton_poison_fit(self, units_y='counts'):
        units_y = 'y_{0}'.format(units_y)

        if units_y.startswith('y_control'):
            start, end = self.get_control_start_end_indices()
            start = start if self.fit_start_end_indices[0] is None else self.fit_start_end_indices[0]
            end = end if self.fit_start_end_indices[1] is None else self.fit_start_end_indices[1]
            x = np.array(self.data['x_control_time_us'][start:end])
        elif units_y.startswith('y_signal'):
            start, end = self.get_signal_start_end_indices()
            start = start if self.fit_start_end_indices[0] is None else self.fit_start_end_indices[0]
            end = end if self.fit_start_end_indices[1] is None else self.fit_start_end_indices[1]
            x = np.array(self.data['x_signal_time_us'][start:end])
        else:
            return None
        y = np.array(self.data[units_y][start:end])

        # we can only do the fitting if there are enough data points
        if len(y) < 4:
            return None

        x0 = float(list(x)[0])

        if self.turton_poison_fit_decay_option == 'single':
            fitmng = exp_with_bg_fit_turton_poison(x - x0, y)
        elif self.turton_poison_fit_decay_option == 'double':
            fitmng = double_exp_with_bg_fit_turton_poison(x - x0, y)

        if self.turton_poison_fit is None:
            self.turton_poison_fit = dict()
        self.turton_poison_fit[units_y] = fitmng

        return fitmng

    def get_binned_data(self, step: int = 2):
        df = self.data
        new_df = DataFrame26(df.default_keys, df.allowed_units, df.spacer)

        for key in df.keys():
            data_list = np.array(df[key])
            if 'time' not in key:
                new_data_list = [sum(data_list[i:i + step]) for i in range(0, len(data_list), step)]
            else:
                new_data_list = [data_list[i] for i in range(0, len(data_list), step)]
            new_df[key] = new_data_list

        return new_df


class DataT1(DataXXX):
    allowed_file_extensions = ['csv']
    allowed_units = {'Time': unit_families['Time']}
    default_keys = ['x_time_us', 'y_counts', 'y_counts_per_cycle', 'y_normalized_counts']
    spacer = '_'
    qdlf_datatype = 't1'

    def __init__(self, file_name, folder_name='.', run_turton_poison_fit_functions=True):

        self.infoT1 = DataDictT1()
        self.size = -1
        self.run_turton_poison_fit_functions = run_turton_poison_fit_functions
        self.turton_poison_fit = None

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer)

    def __post_init__(self):
        self.y_data_in_counts_per_cycle()
        self.y_data_in_normalized_on_peak_counts()

        if self.run_turton_poison_fit_functions:
            try:
                self.get_turton_poison_fit(units_y='counts')
                self.get_turton_poison_fit(units_y='normalized_counts')
                if self.infoT1['numRun'] is not None:
                    self.get_turton_poison_fit(units_y='counts_per_cycle')
            except (ValueError, RuntimeError) as e:
                warnings.warn('Turton-Poison fit was not possible: ' + e)

    def get_data(self):

        file = open(self.folder_name + '/' + self.file_name)
        csv_reader = csv.reader(file)

        self.size = sum(1 for line in csv_reader) - 3
        if self.size < 1:
            self.size = -2
            return False

        file.seek(0)
        x_time = [-9999] * self.size  # initialization
        y_counts = [-1] * self.size  # initialization
        # getting time and counts
        for i in range(self.size):
            line = next(csv_reader)
            x_time[i] = float(line[0])
            y_counts[i] = float(line[1])

        self.data['x_time_us'] = x_time
        self.data['y_counts'] = y_counts
        self.labels['x_time_us'] = 'Time (us)'
        self.labels['y_counts'] = 'Counts (Arb. Units)'

        # getting rest of data
        line = next(csv_reader)
        self.infoT1['numRun'] = float(line[0])
        self.infoT1['pumpOnTime_us'] = float(line[1])

        line = next(csv_reader)
        self.infoT1['gateOnTime_us'] = float(line[0])
        self.infoT1['gateOffsetTime_us'] = float(line[1])

        line = next(csv_reader)
        self.infoT1['numPerRun'] = float(line[0])
        self.infoT1['clockRes_us'] = float(line[1])

        if self.infoT1['numRun'] - x_time[-1] == x_time[-1] - x_time[-2]:
            warnings.warn('This file might be a OP file')

        file.close()
        return True

    def get_additional_info(self):
        return {'infoT1': dict(self.infoT1), 'size': self.size,
                'turton_poison_fit': self.turton_poison_fit,
                'run_turton_poison_fit_functions': self.run_turton_poison_fit_functions}

    def set_additional_info(self, info_dictionary):
        self.infoT1 = DataDictT1(**info_dictionary['infoT1'])
        self.size = info_dictionary['size']
        self.turton_poison_fit = info_dictionary['turton_poison_fit']
        self.run_turton_poison_fit_functions = info_dictionary['run_turton_poison_fit_functions']

    def y_data_in_counts_per_cycle(self):
        rescaling_factor_cycle = (self.infoT1['numRun'] * self.infoT1['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle'] = [counts / rescaling_factor_cycle for counts in self.data.y_counts]
        self.labels['y_counts_per_cycle'] = 'Counts/cycle'
        return True

    def y_data_in_normalized_on_peak_counts(self):
        rescaling_factor_normalize = self.data.y_counts.max()  # normalized
        self.data['y_normalized_counts'] = [counts / rescaling_factor_normalize for counts in self.data.y_counts]
        self.labels['y_normalized_counts'] = 'Counts normalized to max'
        return True

    def get_turton_poison_fit(self, units_y='counts'):
        units_y = 'y_{0}'.format(units_y)

        x = np.array(self.data['x_time_us'])
        y = np.array(self.data[units_y])

        # we can only do the fitting if there are enough data points
        if len(y) < 4:
            return None

        fitmng = exp_with_bg_fit_turton_poison(x, y)

        if self.turton_poison_fit is None:
            self.turton_poison_fit = dict()
        self.turton_poison_fit[units_y] = fitmng

        return fitmng


# Added by Chris Zimmerman
class DataRFSpectrum(DataXXX):
    allowed_file_extensions = ['mat']
    allowed_units = {'Frequency': unit_families['Frequency'], 'Power': unit_families['Power']}
    default_keys = ['x_MHz', 'y_power_ratio_dB', 'y_power_ratio', 'y_voltage_ratio']
    spacer = '_'
    qdlf_datatype = 'rfspectrum'

    def __init__(self, file_name, folder_name='./'):
        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer,
                         self.qdlf_datatype)

        self.resonance_fit = None
        self.quality_factor = None

    def get_data(self):
        data = spio.loadmat(self.folder_name + self.file_name)
        self.data['x_MHz'] = data['freqsweep'][0]
        self.data['y_power_ratio_dB'] = data['powerread'][0]
        self.data['y_power_ratio'] = 10 ** (self.data['y_power_ratio_dB'] / 10)
        self.data['y_voltage_ratio'] = np.sqrt(self.data['y_power_ratio'])

        return True

    def get_additional_info(self):
        return {'resonance_fit': self.resonance_fit, 'quality_factor': self.quality_factor}

    def set_additional_info(self, info_dictionary):
        self.resonance_fit = info_dictionary['resonance_fit']
        self.quality_factor = info_dictionary['quality_factor']

    def calculate_quality_factor(self, initial_width=1):
        def lorentzian(x, amplitude, position, width, offset):
            return amplitude * (width ** 2 / ((x - position) ** 2 + width ** 2)) + offset

        def get_full_width_at_half_maximum(resonance_fit):
            data = resonance_fit.copy()

            data['y_power_ratio'] -= data['y_power_ratio'].max()
            data['y_power_ratio'] = np.abs(data['y_power_ratio'])
            data['delta_y_power_ratio'] = np.abs(data['y_power_ratio'] - data['y_power_ratio'].max() / 2)
            peak_position = data.loc[data['y_power_ratio'] == data['y_power_ratio'].max()].x_MHz.to_numpy()[0]

            data_left = data.loc[data['x_MHz'] <= peak_position]
            data_right = data.loc[data['x_MHz'] >= peak_position]

            fwhm = (data.loc[data_right['delta_y_power_ratio'].idxmin(), 'x_MHz']
                    - data.loc[data_left['delta_y_power_ratio'].idxmin(), 'x_MHz'])
            return fwhm

        initial_offset = self.data['y_power_ratio_dB'].max()
        initial_amplitude = self.data['y_power_ratio_dB'].min()
        initial_position = self.data.loc[self.data['y_power_ratio_dB'] == initial_amplitude].x_MHz.to_numpy()[0]

        p0 = [initial_amplitude, initial_position, initial_width, initial_offset]
        data_for_fitting = self.data.loc[(self.data['x_MHz'] >= initial_position - initial_width)
                                         & (self.data['x_MHz'] <= initial_position + initial_width)]
        parameters, covariance = spo.curve_fit(lorentzian, data_for_fitting['x_MHz'],
                                               data_for_fitting['y_power_ratio_dB'], p0=p0)

        self.resonance_fit = pd.DataFrame(data={'x_MHz': self.data['x_MHz'],
                                                'y_power_ratio_dB': lorentzian(self.data['x_MHz'], parameters[0],
                                                                               parameters[1], parameters[2],
                                                                               parameters[3])})
        self.resonance_fit['y_power_ratio'] = 10 ** (self.resonance_fit['y_power_ratio_dB'] / 10)

        peak_pos = parameters[1]
        fwhm = get_full_width_at_half_maximum(self.resonance_fit)
        self.quality_factor = peak_pos / fwhm

        return True


# Added by Chris Zimmerman
class DataSPCMCounter(DataXXX):
    allowed_file_extensions = ['mat']
    allowed_units = {'Time': unit_families['Time']}
    default_keys = ['time', 'counts_per_second']
    spacer = '_'
    qdlf_datatype = 'spcmcounter'

    def __init__(self, file_name, folder_name='./'):
        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer,
                         self.qdlf_datatype)

    def get_data(self):
        data = spio.loadmat(self.folder_name + self.file_name)
        counts = np.transpose(data['data'][0][0][7][0][0])[0]
        self.data = pd.DataFrame(data={'time': range(len(counts)), 'counts_per_second': counts[::-1]})

        return True


class DataQDLF(DataXXX):
    allowed_file_extensions = ['qdlf', 'in_code_data_manager']

    def __init__(self, file_name, default_keys, allowed_units, spacer, qdlf_datatype, folder_name='./',
                 data_manager=None):
        """In case the user provides a data manager, the filename and folder are neglected"""
        self.average = None
        self.stdev = None
        self.data_manager = data_manager
        super().__init__(file_name, folder_name, default_keys, allowed_units, self.allowed_file_extensions, spacer,
                         qdlf_datatype)

    def __post_init__(self):
        self._set_averages()
        self._set_stdevs()

    def get_additional_info(self):
        return {'average': self.average, 'stdev': self.stdev}

    def set_additional_info(self, info_dictionary):
        self.average = info_dictionary['average']
        self.stdev = info_dictionary['stdev']

    def _set_averages(self):
        self.average = DataFrame26(self.default_keys, self.allowed_units, self.spacer)
        for key in self.data.keys():
            self.average[key] = [np.average(self.data[key])]

    def _set_stdevs(self):
        self.stdev = DataFrame26(self.default_keys, self.allowed_units, self.spacer)
        for key in self.data.keys():
            self.stdev[key] = [np.std(self.data[key])]


class DataSPCM(DataQDLF):
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Time': unit_families['Time']}
    default_keys = ['x_second', 'y_counts', 'y_counts_per_second', 'y_counts_per_second_per_power']
    spacer = '_'
    qdlf_datatype = 'spcm'

    def __init__(self, file_name, folder_name='./', data_manager=None):
        super().__init__(file_name, self.default_keys, self.allowed_units, self.spacer, self.qdlf_datatype, folder_name,
                         data_manager)

    def get_data(self):
        # gets data from raw files
        if self.data_manager is None:
            from .filing.QDLFiling import QDLFDataManager
            self.data_manager = QDLFDataManager.load(self.folder_name + '/' + self.file_name)

        file = self.data_manager
        self.data['x_second'] = file.data['x1']
        self.data['y_counts'] = file.data['y1']
        self.data['y_counts_per_second'] = file.data['y1'] / file.parameters['time_step']

        return True


class DataPower(DataQDLF):
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Time': unit_families['Time']}
    default_keys = ['x_second', 'y_uW']
    spacer = '_'

    qdlf_datatype = 'power'

    def __init__(self, file_name, folder_name='./', data_manager=None, force_power_ratio=False):
        self.force_power_ratio = force_power_ratio
        super().__init__(file_name, self.default_keys, self.allowed_units, self.spacer, self.qdlf_datatype, folder_name,
                         data_manager)

    def get_data(self):
        # gets data from raw files
        if self.data_manager is None:
            from .filing.QDLFiling import QDLFDataManager
            self.data_manager = QDLFDataManager.load(self.folder_name + '/' + self.file_name)

        file = self.data_manager
        self.data['x_second'] = file.data['x1']
        self.data['y_uW'] = file.data['y1']

        if 'x2' in file.data.keys():
            self.data['x2_second'] = file.data['x2']
            self.data['y2_uW'] = file.data['y2']
            if self.force_power_ratio:
                y1 = np.array(file.data['y1'])
                y2 = np.array(file.data['y2'])
                for i in range(len(y1)):
                    if y1[i] > y2[i]:
                        temp = y1[i]
                        y1[i] = y2[i]
                        y2[i] = temp
                self.data['y_uW'] = y1
                self.data['y2_uW'] = y2

        return True


class DataWavelength(DataQDLF):
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Frequency': unit_families['Frequency'], 'Time': unit_families['Time']}
    spacer = '_'
    default_keys = ['x_second', 'y_nm']
    qdlf_datatype = 'wavelength'

    def __init__(self, file_name, folder_name='./', data_manager=None):
        super().__init__(file_name, self.default_keys, self.allowed_units, self.spacer, self.qdlf_datatype, folder_name,
                         data_manager)

    def get_data(self):
        # gets data from raw files
        if self.data_manager is None:
            from .filing.QDLFiling import QDLFDataManager
            self.data_manager = QDLFDataManager.load(self.folder_name + '/' + self.file_name)

        file = self.data_manager
        self.data['x_second'] = file.data['x1']
        self.data['y_nm'] = file.data['y1']

        return True



