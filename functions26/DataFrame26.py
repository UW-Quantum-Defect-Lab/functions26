# 2020-09-12
# This code was made for use in the Fu lab
# by Vasilis Niaouris

from pandas import DataFrame
import warnings

# DataFrame26 is a pandas DataFrame spin-off for the 26 room.
# We want to be able to change the units of the data without saving the data in all the different possible units
# We also want to avoid the user adding columns they are not supposed to (without them being aware they do so).


# get_unit receives a key and a dict of a dict of units and checks if the last part of the key belongs in the unit dict
def get_unit(key, allowed_units, spacer):
    key_unit = get_unit_string(key, spacer)
    for unit_family in allowed_units:
        for unit in allowed_units[unit_family]:
            if unit == key_unit:
                return key_unit, unit_family

    return None, None


# get_dataframe_column checks if all other than the last part (before last '_') is part of a key that belongs to
# the dataframe_columns
def get_dataframe_column(key, dataframe_columns, allowed_units, unit_family, spacer):
    key_initial = get_init_string(key, spacer)
    for column_string in dataframe_columns:
        if key_initial == get_init_string(column_string, spacer) \
                and get_unit_string(column_string, spacer) in allowed_units[unit_family]:
            return key_initial, column_string

    return None, None


def get_init_string(key, spacer):
    string_list = key.split(spacer)[:-1]
    return ' '.join(string_list)


def get_unit_string(key, spacer):
    string_list = key.split(spacer)[-1]  # split is redundant for '_' style, but important for ' ' spacer
    return string_list.strip('()')  # the [0] is because it will return a list with a single string in it


warnings.simplefilter('ignore', UserWarning)


class DataFrame26(DataFrame):

    # this function is called when initializing the object e.g. dataframe26object = Dataframe26()
    # the modified init, checks if we gave the default_keys and allowed_units
    # allowed_units example: allowed_units = {'Length': {'nm': 1., 'um': 1.e-3}, 'Energy': {'eV': 1., 'meV': 1.e3}}
    # default_keys example: default_keys = ['x_nm', 'x_eV']
    def __init__(self, default_keys=None, allowed_units=None, spacer=None, restrict_to_defaults=False, qdlf_datatype=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if default_keys is None:
            self.default_keys = []
        else:
            self.set_default_keys(default_keys)
        if allowed_units is None:
            self.allowed_units = {}
        else:
            self.set_allowed_units(allowed_units)
        if spacer == '_' or spacer == ' ':
            self.spacer = spacer
        elif spacer is None:
            warnings.warn('DataFrame26 spacer was not given. Set to default: _')
            self.spacer = '_'
        else:
            warnings.warn('DataFrame26 spacer not valid. Set to default: _')
            self.spacer = '_'
        self.restrict_to_defaults = restrict_to_defaults
        self.qdlf_datatype = qdlf_datatype

    # this function is called if you for example call: dataframe26object[key] = [3,4] (have to assign values)
    # the modified setitem, makes sure we are not adding any keys not listed in default_keys
    def __setitem__(self, key, value):
        if self.restrict_to_defaults and key not in self.default_keys and key not in self.keys():
            # second part of if statements is here in case you change default_keys and old keys are not in them
            raise RuntimeError("Appending keys not listed in default_keys of type<DataFrame26> is not allowed")
        else:
            super(DataFrame26, self).__setitem__(key, value)

    # this function is called if you for example call: dataframe26object[key] without assigning any values
    # the modified getitem checks if the key we requested exists in different units.
    # if it does, it returns a pandas.core.series.Series object (as the original code)
    # but it does not save it in the Dataframe26 object.
    # for unknown reason when we call
    def __getitem__(self, key):
        try:
            if key not in self.default_keys and key not in self.keys():
                # second part of if statements is here in case you change default_keys and old keys are not in them
                key_unit, key_unit_family = get_unit(key, self.allowed_units, self.spacer)  # string
                key_column, original_column_key = get_dataframe_column(key, self.keys(),
                                                                       self.allowed_units, key_unit_family,
                                                                       self.spacer)  # string
                if key_unit is not None and key_column is not None:
                    data = super(DataFrame26, self).__getitem__(original_column_key)
                    original_column_key_units = self.allowed_units[key_unit_family][get_unit_string(original_column_key,
                                                                                                    self.spacer)]
                    # actually add this new unit to the dataframe26
                    if not self.restrict_to_defaults:
                        super(DataFrame26, self).__setitem__(key, data.rename(key)
                                                             * self.allowed_units[key_unit_family][key_unit]
                                                             / original_column_key_units)
                        return data.rename(key) * self.allowed_units[key_unit_family][key_unit] / original_column_key_units
                else:
                    return super(DataFrame26, self).__getitem__(key)
        except AttributeError:
            pass

        return super(DataFrame26, self).__getitem__(key)

    # def get_loc(self, key, method=None, tolerance=None):
    #     print(self.__getitem__(key))
    #     return super(DataFrame26, self).get_loc(key, method, tolerance)

    # A function to help change the default_keys if necessary
    def set_default_keys(self, default_keys):
        if isinstance(default_keys, list):
            if all(isinstance(item, str) for item in default_keys) or len(default_keys) == 0:
                # after or we check for empty lists
                with warnings.catch_warnings():  # suppressing the 'column can not be created with an attribute' warning
                    warnings.simplefilter('ignore', UserWarning)
                    self.default_keys = default_keys
                return True

        raise TypeError('Default keys must be given in a list')
        # if the function doesnt return true, then we get an error

    def get_default_keys(self):
        return self.default_keys

    # A function to help change the allowed_units if necessary
    def set_allowed_units(self, allowed_units):
        if isinstance(allowed_units, dict):
            if all(isinstance(allowed_units[unit_family], dict) for unit_family in allowed_units):
                if all((isinstance(allowed_units[unit_family][unit], str) for unit in unit_family)
                       for unit_family in allowed_units):
                    with warnings.catch_warnings():  # suppressing the 'column can not be created with
                        # an attribute' warning
                        warnings.simplefilter('ignore', UserWarning)
                        self.allowed_units = allowed_units
                    return True

        raise TypeError('Allowed units must be given in a dictionary of a dictionary')
        # if the function doesn't return true, then we get an error

    def get_allowed_units(self):
        return self.allowed_units

    def save_with_qdlf_data_manager(self, filename, qdlf_parameters_dict=None, qdlf_datatype=None):
        from .filing.QDLFiling import QDLFDataManager
        if qdlf_datatype is None:
            qdlf_datatype = self.qdlf_datatype
        filedm = QDLFDataManager(data=self, parameters=qdlf_parameters_dict, datatype=qdlf_datatype)
        filedm.save(filename)

# Import DataFrame26 and run next lines to see how it works
# ee = DataFrame26(data={'x_nm': [2, 3]}, default_keys=['x_nm', 'x_eV'],
#                  allowed_units={'Length': {'nm': 1., 'um': 1.e-3}, 'Energy': {'eV': 1., 'meV': 1.e3}})
#
# print(ee['x_um'])
# ee['x_eV'] = [34, 56]
# print(ee['x_meV'])
