# 2020-09-10
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Added/Edited by Chris on 2020-09-21

import warnings

from .Dict26 import Dict26
from .units import unit_families
from .units.UnitClass import UnitClass, UnitClassList


def convert_file_string_to_float(string, key_unit, allowed_units):
    # Format: XpYunit = X.Y unit + key_unit[-1] or X = X unit + key_unit[-1]
    unit_multiplier = 1.
    if not string[-1].isdigit():
        string_list = list(string)
        i = len(string_list) - 1
        string_unit_list = []
        while i >= 0:
            if not string[i].isdigit():
                string_unit_list.insert(0, string[i])
                i -= 1
            else:
                break

        # get string without units
        string = ''.join(string_list[:len(string_list)-len(string_unit_list)])
        # get units from string and from key
        string_unit = ''.join(string_unit_list) + key_unit[-1]

        # find if in unit families and
        for family in allowed_units:
            for unit in allowed_units[family]:
                if string_unit == unit:
                    unit_multiplier = allowed_units[family][key_unit] / allowed_units[family][string_unit]

        if unit_multiplier == 1. and string_unit != key_unit:
            warnings.warn('in convert_file_string_to_float -> unit string: (' + string_unit[:-1] + ') is not recognized')

    if 'p' in string:
        return float(string.replace('p', '.')) * unit_multiplier
    else:
        return float(string) * unit_multiplier


class DataDictSpectrum(Dict26):
    default_keys = ['cal_data', 'exposure_time_secs', 'cycles',
                    'wavelength_offset_nm', 'background_counts_per_second',
                    'background_counts_per_cycle', 'background_counts']
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Time': unit_families['Time']}

    def __init__(self, *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer='_', *args, **kwargs)


class DataDictSIF(Dict26):
    default_keys = ['cal_data', 'exposure_time_secs', 'cycles',
                    'wavelength_offset_nm', 'background_counts_per_second',
                    'background_counts_per_cycle', 'background_counts']
    allowed_units = {'Length': unit_families['Length'], 'Energy': unit_families['Energy'],
                     'Time': unit_families['Time']}

    def __init__(self, *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer='_', *args, **kwargs)


class DataDictOP(Dict26):
    default_keys = ['numRun', 'numPerRun', 'pumpOnTime_us', 'pumpOffTime_us']
    allowed_units = {'Time': unit_families['Time']}

    def __init__(self, *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer='_', *args, **kwargs)


class DataDictOP2LaserDelay(Dict26):
    default_keys = ['numRun', 'numPerRun', 'controlPumpOnTime_us', 'signalPumpOnTime_us', 'controlSignalDelayTime_us',
                    'pumpOffTime_us']
    allowed_units = {'Time': unit_families['Time']}

    def __init__(self, *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer='_', *args, **kwargs)


class DataDictT1(Dict26):
    default_keys = ['numRun', 'numPerRun', 'pumpOnTime_us', 'gateOnTime_us', 'gateOffsetTime_us', 'clockRes_us']
    allowed_units = {'Time': unit_families['Time']}

    def __init__(self, *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer='_', *args, **kwargs)


class DataDictLaserInfo(Dict26):
    default_keys_with_acronyms = {'Type': 'type',
                                  'Wavelength (nm)': 'wavelength',
                                  'Power (nW)': 'power'}
    default_keys = [key for key in default_keys_with_acronyms]
    allowed_units = {'Length': unit_families['Length'], 'Power' : unit_families['Power']}

    default_head_keys = ['Laser',
                         'Secondary Laser']

    def __init__(self, head_key='Laser', *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer=' ',
                         restrict_to_defaults=True, *args, **kwargs)
        if head_key in self.default_head_keys:
            self.head_key = head_key
            for key in self.default_keys:
                self.setdefault(key, None)
        else:
            raise RuntimeError('Given head_key cannot be found in default_head_keys list')

    def __setitem__(self, key, value):
        super(DataDictLaserInfo, self).__setitem__(key, value)
        unit = self.get_units(key)
        if unit is None:
            setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '), value)
        else:
            if isinstance(value, list):
                setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '),
                        UnitClassList(value, unit))
            else:
                setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '),
                        UnitClass(value, unit))

    def get_units(self, key):
        for family in self.allowed_units:
            for unit in self.allowed_units[family]:
                if key.split(' ')[-1].strip('()') == unit:
                    return unit

        return None

    def get_info(self, string):
        if string is None:
            return False
        info_string_components = string.split('-')
        for info, key in zip(info_string_components, self.default_keys):
            if self.get_units(key) is None:
                self[key] = info
            elif info[:5] == 'sweep':
                self[key] = [convert_file_string_to_float(info.split('to')[0][5:],
                                                          self.get_units(key), self.allowed_units),
                             convert_file_string_to_float(info.split('to')[1].split('step')[0],
                                                          self.get_units(key), self.allowed_units),
                             convert_file_string_to_float(info.split('to')[1].split('step')[1],
                                                          self.get_units(key), self.allowed_units)]
            else:
                self[key] = convert_file_string_to_float(info, self.get_units(key), self.allowed_units)
        return True


class DataDictRFSourceInfo(Dict26):
    default_keys_with_acronyms = {'Type': 'type',
                                  'Frequency (MHz)': 'frequency',
                                  'Power (dBm)': 'power',
                                  'Circuit components': 'circuit_components'}
    default_keys = [key for key in default_keys_with_acronyms]
    allowed_units = {'Frequency': unit_families['Frequency'], 'Power': unit_families['Power']}

    default_head_keys = ['RF Source']

    def __init__(self, head_key='RF Source', *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer=' ',
                         restrict_to_defaults=True, *args, **kwargs)
        if head_key in self.default_head_keys:
            self.head_key = head_key
            for key in self.default_keys:
                self.setdefault(key, None)
        else:
            raise RuntimeError('Given head_key cannot be found in default_head_keys list')

    def __setitem__(self, key, value):
        super(DataDictRFSourceInfo, self).__setitem__(key, value)
        unit = self.get_units(key)
        if unit is None:
            setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '), value)
        else:
            if isinstance(value, list):
                setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '),
                        UnitClassList(value, unit))
            else:
                setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '),
                        UnitClass(value, unit))

    def get_units(self, key):
        for family in self.allowed_units:
            for unit in self.allowed_units[family]:
                if key.split(' ')[-1].strip('()') == unit:
                    return unit

        return None

    def get_info(self, string):
        if string is None:
            return False
        info_string_components = string.split('-')
        for info, key in zip(info_string_components, self.default_keys):
            if self.get_units(key) is None:
                self[key] = info
            elif info[:5] == 'sweep':
                self[key] = [convert_file_string_to_float(info.split('to')[0][5:],
                                                          self.get_units(key), self.allowed_units),
                             convert_file_string_to_float(info.split('to')[1].split('step')[0],
                                                          self.get_units(key), self.allowed_units),
                             convert_file_string_to_float(info.split('to')[1].split('step')[1],
                                                          self.get_units(key), self.allowed_units)]
            else:
                self[key] = convert_file_string_to_float(info, self.get_units(key), self.allowed_units)
        return True


class DataDictPathOpticsInfo(Dict26):
    default_keys_with_acronyms = {'Half-Waveplate Angle (deg)': 'WP2',
                                  'Quarter-Waveplate Angle (deg)': 'WP4',
                                  'Polarizer': 'Plr',
                                  'Pinhole': 'PnH',
                                  'Filter': 'Flt'}
    default_keys = [key for key in default_keys_with_acronyms]
    allowed_units = {'Angle': unit_families['Angle']}

    default_head_keys = ['Excitation Path Optics',
                         'Collection Path Optics',
                         'Excitation and Collection Path Optics']

    def __init__(self, head_key, *args, **kwargs):
        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer=' ',
                         restrict_to_defaults=True, *args, **kwargs)
        if head_key in self.default_head_keys:
            self.head_key = head_key
            for key in self.default_keys:
                self.setdefault(key, None)
        else:
            raise RuntimeError('Given head_key cannot be found in default_head_keys list')

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        unit = self.get_units(key)
        if unit is None:
            setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '), value)
        else:
            if isinstance(value, list):
                setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '),
                        UnitClassList(value, unit))
            else:
                setattr(self, self.default_keys_with_acronyms[key].lower().strip(' '),
                        UnitClass(value, unit))

    def get_units(self, key):
        for family in self.allowed_units:
            for unit in self.allowed_units[family]:
                if key.split(' ')[-1].strip('()') == unit:
                    return unit

        return None

    def get_info(self, string):
        if string is None:
            return False
        info_string_components = string.split('-')
        for key in self.default_keys_with_acronyms:
            for info in info_string_components:
                if info.startswith(self.default_keys_with_acronyms[key]):
                    info = info.split('{0}~'.format(self.default_keys_with_acronyms[key]))[1]
                    if self.get_units(key) is None:
                        self[key] = info
                    else:
                        self[key] = convert_file_string_to_float(info, self.get_units(key), self.allowed_units)
                    break
        return True


class DataDictFilenameInfo(Dict26):
    default_main_keys_with_acronyms = {'File Number': 'FNo',
                                       'Sample Name': 'Smp',
                                       'Laser': 'Lsr',
                                       'Secondary Laser': 'Ls2',
                                       'RF Source': 'RFS',
                                       'Magnetic Field (T)': 'MgF',
                                       'Temperature (K)': 'Tmp',
                                       'Measurement Type': 'MsT',
                                       'Excitation Path Optics': 'Exc',
                                       'Collection Path Optics': 'Col',
                                       'Excitation and Collection Path Optics': 'EnC',
                                       'Miscellaneous': 'Msc'}

    # getting additional information dictionary
    lsr_info = DataDictLaserInfo('Laser')
    ls2_info = DataDictLaserInfo('Secondary Laser')
    rfs_info = DataDictRFSourceInfo('RF Source')
    exc_info = DataDictPathOpticsInfo('Excitation Path Optics')
    col_info = DataDictPathOpticsInfo('Collection Path Optics')
    enc_info = DataDictPathOpticsInfo('Excitation and Collection Path Optics')
    file_additional_info_list = [lsr_info, ls2_info, rfs_info, exc_info, col_info, enc_info]
    # returning head_key dict for all fai
    fai_head_keys_dict = {fai.head_key: fai for fai in file_additional_info_list}

    allowed_units = unit_families
    key_sub_key_separator = ': '

    def __init__(self, *args, **kwargs):

        self.default_keys = []
        self.reset_default_keys()

        super().__init__(default_keys=self.default_keys, allowed_units=self.allowed_units, spacer=' ',
                         restrict_to_defaults=True, *args, **kwargs)

        for key in self.default_keys:
            self.setdefault(key, None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # to avoid defining two attributes twice, one here and one embedded
        if key.split(self.key_sub_key_separator)[0] not in self.default_main_keys_with_acronyms.values():
            # For keys under additional info dictinaries, we set the dictionary to the attribute (see bellow)
            if key not in self.fai_head_keys_dict.keys():
                unit = self.get_units(key)
                if unit is None:
                    setattr(self, self.default_main_keys_with_acronyms[key].lower().strip(' '), value)
                else:
                    if isinstance(value, list):
                        setattr(self, self.default_main_keys_with_acronyms[key].lower().strip(' '),
                                UnitClassList(value, unit))
                    else:
                        setattr(self, self.default_main_keys_with_acronyms[key].lower().strip(' '),
                                UnitClass(value, unit))
        else:
            # Here we want to set the attribute of an additional info category (key)
            # first we get the header attribute by simply looking at he first part of the key
            headattr_string = key.split(self.key_sub_key_separator)[0].lower().strip(' ')
            # then we search the value of the first part of the hkey in the default keys dictionary and find its
            # corresponding key which is the head_key of the additional info dictionary
            head_key = list(self.default_main_keys_with_acronyms.keys())[
                list(self.default_main_keys_with_acronyms.values()).index(key.split(self.key_sub_key_separator)[0])]
            sub_key = key.split(self.key_sub_key_separator)[1]

            try:
                getattr(self, headattr_string)[sub_key] = value
            except AttributeError:
                # Check what kind of class fai with this head key is
                class_to_init = type(self.fai_head_keys_dict[head_key])
                # Create an attribute of this type.
                setattr(self, headattr_string, class_to_init(head_key))
                # Add the new value
                getattr(self, headattr_string)[sub_key] = value

        return True

    def get_units(self, key):
        for family in self.allowed_units:
            for unit in self.allowed_units[family]:
                if key.split(' ')[-1].strip('()') == unit:
                    return unit

        return None

    def reset_default_keys(self):
        main_default_keys = [key for key in self.default_main_keys_with_acronyms]
        additional_default_keys = []
        for fhk in self.fai_head_keys_dict:
            for sub_def_key in self.fai_head_keys_dict[fhk].default_keys:
                additional_default_keys.append(self.get_sub_key_string(fhk, sub_def_key))

        self.default_keys = main_default_keys + additional_default_keys

        return True

    def get_sub_key_string(self, head_key, sub_key):
        return self.default_main_keys_with_acronyms[head_key] + self.key_sub_key_separator + sub_key

    def get_info(self, file_info_raw_components):
        try:
            self['File Number'] = int(file_info_raw_components[0])  # Filename starts with file number
        except:
            warnings.warn('Filename does not follow filenaming convention')
        # Loop through the components of the file_name and check which of the default_main_keys_with_acronyms
        # they contain
        # Extract the corresponding info and save
        # If some key is not part of the file name, set corresponding entry to None
        try:
            for firc in file_info_raw_components[1:]:
                for dmk in self.default_main_keys_with_acronyms:
                    # getting main info
                    try:
                        if firc.startswith(self.default_main_keys_with_acronyms[dmk]):
                            info = firc.split('{0}~'.format(self.default_main_keys_with_acronyms[dmk]))[1]
                            if self.get_units(dmk) is None:
                                self[dmk] = info
                            else:
                                self[dmk] = convert_file_string_to_float(info, self.get_units(dmk), self.allowed_units)
                            break
                        # Some keys can contain more than one piece of information
                        # for example, Lsr~cw-720 -> Type = cw,  Wavelength (nm) = 720, and Power (nW) = 100
                        # getting additional info in separate keys
                        if dmk in self.fai_head_keys_dict.keys():
                            self.fai_head_keys_dict[dmk].get_info(self[dmk])
                            for sub_key in self.fai_head_keys_dict[dmk]:
                                self[self.get_sub_key_string(dmk, sub_key)] = self.fai_head_keys_dict[dmk][sub_key]
                    except:
                        warnings.warn('Filename does not follow filenaming convention')
        except:
            warnings.warn('Filename does not follow filenaming convention')
        return True
