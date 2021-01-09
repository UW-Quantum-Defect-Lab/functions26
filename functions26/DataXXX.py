# 2019-11-19 and last update on 2020-09-14
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# DataSIF modified by Christian Zimmermann

from sif_reader import np_open as readsif
import csv
import numpy as np
from .constants import conversion_factor_nm_to_ev  # eV*nm
from .DataFrame26 import DataFrame26
from .Dict26 import Dict26
from .units import unit_families
from .DataDictXXX import DataDictSIF, DataDictOP, DataDictT1
from .DataDictXXX import DataDictFilenameInfo, DataDictLaserInfo, DataDictPathOpticsInfo
from .useful_functions import get_added_label_from_unit
import warnings


class DataXXX:

    def __init__(self, file_name, folder_name, default_keys, allowed_units, allowed_file_extensions, spacer):
        self.file_name = file_name
        if file_name == '':
            raise RuntimeError('File name is an empty string')
        self.folder_name = folder_name
        self.file_extension = self.file_name.split('.')[-1]
        self.check_file_type(allowed_file_extensions)

        self.file_info = DataDictFilenameInfo()
        self.get_file_info()

        self.data = DataFrame26(default_keys, allowed_units, spacer)
        self.labels = Dict26(default_keys, allowed_units, spacer)
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


class DataSIF(DataXXX):

    allowed_file_extensions = ['sif']
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Time': unit_families['Time']}
    default_keys = ['x_pixel', 'x_nm', 'x_eV',
                    'y_counts', 'y_counts_per_cycle',
                    'y_counts_per_second', 'y_counts_per_second_per_power',
                    'y_counts_per_cycle',
                    'y_nobg_counts_per_second', 'y_nobg_counts_per_second_per_power']
    spacer = '_'

    def __init__(self, file_name, second_order=False, wavelength_offset=0, background_per_cycle=300,
                 folder_name='.', from_video=True):

        self.from_video = from_video
        self.second_order = second_order

        # For some reason this doesn't work. it has to do with where super() is called in Dict26, but it is a mystery
        # self.infoSIF = DataDictSIF(data={'wavelength_offset_nm': wavelength_offset,
        #                                  'background_per_cycle': background_per_cycle})
        self.infoSIF = DataDictSIF()
        self.infoSIF['wavelength_offset_nm'] = wavelength_offset
        self.infoSIF['background_counts_per_cycle'] = background_per_cycle

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer)

        self.set_background()
        self.set_all_y_data()
        self.set_all_x_data()
        self.total_counts = self.integrate_counts()  # background is taken into account

    def get_data(self):
        counts_info, acquisition_info = readsif(self.folder_name + '/' + self.file_name)
        counts_extended_list = counts_info.tolist()
        y_counts = counts_extended_list[0][0]
        self.data['y_counts'] = y_counts

        x_pixel = np.array([pixel for pixel in range(1, len(y_counts)+1, 1)])
        self.data['x_pixel'] = x_pixel

        self.infoSIF['cal_data'] = acquisition_info['Calibration_data']
        self.infoSIF['exposure_time_secs'] = acquisition_info['ExposureTime']
        if not self.from_video:
            self.infoSIF['cycles'] = acquisition_info['AccumulatedCycles']
        else:
            self.infoSIF['cycles'] = 1

        self.labels['x_pixel'] = 'Pixel'
        self.labels['y_counts'] = 'Counts'

        return True

    def get_wavelength_calibration(self):
        return self.infoSIF['cal_data'][0] \
               + self.infoSIF['cal_data'][1] * self.data.x_pixel \
               + self.infoSIF['cal_data'][2] * self.data.x_pixel ** 2 \
               + self.infoSIF['cal_data'][3] * self.data.x_pixel ** 3 \
               + self.infoSIF['wavelength_offset_nm']

    def integrate_counts(self, unit='counts', subtract_bg=False):
        tot = self.data['y_{0}'.format(unit)].sum()
        if subtract_bg:
            tot -= len(self.data)*self.infoSIF['background_{0}'.format(unit)]
        return tot

    def integrate_in_region(self, start, end, unit_x='pixel', unit_y='counts', subtract_bg=False):
        data_in_region = self.data.loc[(self.data['x_{0}'.format(unit_x)] >= start)
                                       & (self.data['x_{0}'.format(unit_x)] <= end)]
        if subtract_bg:
            data_in_region -= self.infoSIF['background_{0}'.format(unit_y)]
        return data_in_region['y_{0}'.format(unit_y)].sum()

    def shift_counts(self, shift_y, unit='counts'):
        self.data['y_{0}'.format(unit)] += shift_y
        return True

    def renormalize_counts(self, times_renorm_value, unit='counts',
                           added_string_to_unit='_renormalized', added_string_to_label=None):

        if added_string_to_label is None:
            added_string_to_label = get_added_label_from_unit(added_string_to_unit)

        self.data['y_{0}'.format(unit+added_string_to_unit)] = self.data['y_{0}'.format(unit)] * times_renorm_value
        self.labels['y_{0}'.format(unit + added_string_to_unit)] = self.labels['y_{0}'.format(unit)] \
                                                                   + added_string_to_label
        return True

    def renormalize_counts_to_max(self, unit='counts', added_string_to_unit='_renormalized_to_max',
                                  added_string_to_label='Renormalized to max'):
        factor = 1./self.data['y_{0}'.format(unit)].max()
        self.renormalize_counts(factor, unit, added_string_to_unit, added_string_to_label)

        return True

    def set_x_data_in_nm(self):
        self.data['x_nm'] = self.get_wavelength_calibration()
        self.labels['x_nm'] = 'Wavelength (nm)'
        return True

    def set_x_data_in_nm_2nd_order(self):
        self.data['x_nm'] = self.get_wavelength_calibration()/2.
        self.labels['x_nm'] = 'Wavelength (nm)'
        return True

    def set_x_data_in_ev(self):
        self.data['x_eV'] = conversion_factor_nm_to_ev/ self.get_wavelength_calibration()
        self.labels['x_eV'] = 'Photon Energy (eV)'
        return True

    def set_x_data_in_ev_2nd_order(self):
        self.data['x_eV'] = conversion_factor_nm_to_ev/(self.get_wavelength_calibration()/2.)
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
        self.infoSIF['background_counts_per_second'] = self.infoSIF['background_counts_per_cycle'] / \
                                                       self.infoSIF['exposure_time_secs'] / self.infoSIF['cycles']

        self.infoSIF['background_counts'] = self.infoSIF['background_counts_per_cycle'] * self.infoSIF['cycles']

        power = self.file_info['Lsr: Power (nW)']
        try:
            self.infoSIF['background_counts_per_second_per_power'] = self.infoSIF['background_counts_per_second'] / \
                                                                     power
        except (ValueError, TypeError):
            warnings.warn('No power information found in file_info. Y data per power were not calculated.')

        return True

    def set_y_data_counts_per_second(self):
        self.data['y_counts_per_second'] = self.data.y_counts/self.infoSIF['exposure_time_secs'] / \
                                           self.infoSIF['cycles']

        self.labels['y_counts_per_second'] = 'Counts/sec'
        return True

    def set_y_data_counts_per_cycle(self):
        self.data['y_counts_per_cycle'] = self.data.y_counts/self.infoSIF['cycles']

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
                                                - self.infoSIF['background_counts_per_second']

        self.labels['y_nobg_counts_per_second'] = 'Counts/sec'
        return True

    def set_y_data_nobg_counts_per_cycle(self):
        self.data['y_nobg_counts_per_cycle'] = self.data['y_counts_per_cycle'] \
                                                - self.infoSIF['background_counts_per_cycle']

        self.labels['y_nobg_counts_per_cycle'] = 'Counts/cycle'
        return True

    def set_y_data_nobg_counts_per_second_per_power(self):
        try:
            self.data['y_nobg_counts_per_second_per_power'] = self.data['y_counts_per_second_per_power'] \
                                                              - self.infoSIF['background_counts_per_second_per_power']
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


class DataOP(DataXXX):

    allowed_file_extensions = ['csv']
    allowed_units = {'Time': unit_families['Time']}
    default_keys = ['x_time_us', 'y_counts', 'y_counts_per_cycle', 'y_normalized_counts']
    spacer = '_'

    def __init__(self, file_name, folder_name='.'):

        self.infoOP = DataDictOP()
        self.size = -1

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer)

        self.y_data_in_counts_per_cycle()
        self.y_data_in_normalized_on_peak_counts()

    def get_data(self):

        file = open(self.folder_name + '/' + self.file_name)
        csv_reader = csv.reader(file)

        self.size = sum(1 for line in csv_reader) - 2
        if self.size < 3:
            self.size = -2  # means there is no data, only run values
            return False

        file.seek(0)
        x_time = [-9999]*self.size  # initialization
        y_counts = [-1]*self.size  # initialization
        # getting time and counts
        for i in range(self.size):
            line = next(csv_reader)
            x_time[i] = float(line[0])
            y_counts[i] = float(line[1])

        self.data['x_time_us'] = x_time
        self.data['y_counts'] = y_counts
        self.labels['x_time_us'] = 'Time (us)'
        self.labels['y_counts'] = 'Counts (Arb. Units)'

        # getting rest info
        line = next(csv_reader)
        self.infoOP['numRun'] = float(line[0])
        self.infoOP['numPerRun'] = float(line[1])

        line = next(csv_reader)
        self.infoOP['pumpOnTime_us'] = float(line[0])
        self.infoOP['pumpOffTime_us'] = float(line[1])

        if x_time[-1]-x_time[-2] != x_time[-2]-x_time[-3]:
            warnings.warn('This file might be a T1 file')

        file.close()
        return True

    def y_data_in_counts_per_cycle(self):
        rescaling_factor_cycle = (self.infoOP['numRun']*self.infoOP['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle'] = [counts/rescaling_factor_cycle for counts in self.data.y_counts]
        self.labels['y_counts_per_cycle'] = 'Counts/cycle'
        return True

    def y_data_in_normalized_on_peak_counts(self):
        rescaling_factor_normalize = self.data.y_counts.max()  # normalized
        self.data['y_normalized_counts'] = [counts/rescaling_factor_normalize for counts in self.data.y_counts]
        self.labels['y_normalized_counts'] = 'Counts normalized to max'
        return True


class DataT1(DataXXX):

    allowed_file_extensions = ['csv']
    allowed_units = {'Time': unit_families['Time']}
    default_keys = ['x_time_us', 'y_counts', 'y_counts_per_cycle', 'y_normalized_counts']
    spacer = '_'

    def __init__(self, file_name, folder_name='.'):

        self.infoT1 = DataDictT1()
        self.size = -1

        super().__init__(file_name, folder_name,
                         self.default_keys, self.allowed_units, self.allowed_file_extensions, self.spacer)

        self.y_data_in_counts_per_cycle()
        self.y_data_in_normalized_on_peak_counts()

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

        if self.infoT1['numRun']-x_time[-1] == x_time[-1]-x_time[-2]:
            warnings.warn('This file might be a OP file')

        file.close()
        return True

    def y_data_in_counts_per_cycle(self):
        rescaling_factor_cycle = (self.infoT1['numRun']*self.infoT1['numPerRun'])  # per cycle
        self.data['y_counts_per_cycle'] = [counts/rescaling_factor_cycle for counts in self.data.y_counts]
        self.labels['y_counts_per_cycle'] = 'Counts/cycle'
        return True

    def y_data_in_normalized_on_peak_counts(self):
        rescaling_factor_normalize = self.data.y_counts.max()  # normalized
        self.data['y_normalized_counts'] = [counts/rescaling_factor_normalize for counts in self.data.y_counts]
        self.labels['y_normalized_counts'] = 'Counts normalized to max'
        return True
