# 2020-09-12
# This code was made for use in the Fu lab
# by Vasilis Niaouris

from collections import UserDict


# Dict26 is a pandas Dictionary spin-off for the 26 room.
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


# get_dictionary_key checks if all other than the last part (before last '_') is part of a key that belongs to
# the dataframe_columns
def get_dictionary_key(key, dictionary_keys, allowed_units, unit_family, spacer):
    key_initial = get_init_string(key, spacer)
    for dict_key in dictionary_keys:
        if key_initial == get_init_string(dict_key, spacer) \
                and get_unit_string(dict_key, spacer) in allowed_units[unit_family]:
            return key_initial, dict_key

    return None, None


def get_init_string(key, spacer):
    string_list = key.split(spacer)[:-1]
    return ' '.join(string_list)


def get_unit_string(key, spacer):
    string_list = key.split(spacer)[-1]  # split is redundant for '_' style, but important for ' ' spacer
    return string_list.strip('()')  # the [0] is because it will return a list with a single string in it


class Dict26(UserDict):

    # this function is called when initializing the object e.g. Dict26object = Dict26()
    # the modified init, checks if we gave the default_keys and allowed_units
    # allowed_units example: allowed_units = {'Length': {'nm': 1., 'um': 1.e-3}, 'Energy': {'eV': 1., 'meV': 1.e3}}
    # default_keys example: default_keys = ['x_nm', 'x_eV']
    def __init__(self, default_keys=None, allowed_units=None, spacer=None, restrict_to_defaults=False, *args, **kwargs):
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
        else:
            print('Dict26 Spacer was not valid or not given. Set to default: _. ')
            self.spacer = '_'
        self.restrict_to_defaults = restrict_to_defaults

    # this function is called if you for example call: Dict26object[key] = [3,4] (have to assign values)
    # the modified setitem, makes sure we are not adding any keys not listed in default_keys
    def __setitem__(self, key, value):
        if self.restrict_to_defaults and key not in self.default_keys and key not in self.keys():
            # second part of if statements is here in case you change default_keys and old keys are not in them
            raise RuntimeError("Appending keys not listed in default_keys of type<Dict26> is not allowed")
        else:
            super(Dict26, self).__setitem__(key, value)

    # this function is called if you for example call: Dict26object[key] without assigning any values
    # the modified getitem checks if the key we requested exists in different units.
    # if it does, it returns a pandas.core.series.Series object (as the original code)
    # but it does not save it in the Dict26 object.
    # for unknown reason when we call
    def __getitem__(self, key):
        if key not in self.default_keys and key not in self.keys():
            # second part of if statements is here in case you change default_keys and old keys are not in them
            key_unit, key_unit_family = get_unit(key, self.allowed_units, self.spacer)  # string
            key_column, original_dictionary_key = get_dictionary_key(key, self.keys(),
                                                                     self.allowed_units, key_unit_family,
                                                                     self.spacer)  # string
            if key_unit is not None and key_column is not None:
                data = super(Dict26, self).__getitem__(original_dictionary_key)
                if not isinstance(data, str):
                    original_column_key_units = self.allowed_units[key_unit_family][get_unit_string(
                        original_dictionary_key, self.spacer)]
                    if isinstance(data, list):
                        if not self.restrict_to_defaults:
                            super(Dict26, self).__setitem__(key, [d * self.allowed_units[key_unit_family][key_unit]
                                                                  / original_column_key_units for d in data])
                        return [d * self.allowed_units[key_unit_family][key_unit] / original_column_key_units
                                for d in data]
                    else:
                        if not self.restrict_to_defaults:
                            super(Dict26, self).__setitem__(key, data * self.allowed_units[key_unit_family][key_unit]
                                                            / original_column_key_units)
                        return data * self.allowed_units[key_unit_family][key_unit] / original_column_key_units
                else:
                    if not self.restrict_to_defaults:
                        super(Dict26, self).__setitem__(key, ' '.join(
                            [get_init_string(self.get(original_dictionary_key), ' '), '(' + key_unit + ')']))
                    return ' '.join([get_init_string(self.get(original_dictionary_key), ' '), '(' + key_unit + ')'])
            else:
                return super(Dict26, self).__getitem__(original_dictionary_key)

        return super(Dict26, self).__getitem__(key)

    # The next three functions are defined this way to avoid deleting keys.
    def __delitem__(self, key):
        raise RuntimeError("Deletion in type<Dict26> not allowed")

    def pop(self, s=None):
        raise RuntimeError("Deletion in type<Dict26> not allowed")

    def popitem(self, s=None):
        raise RuntimeError("Deletion in type<Dict26> not allowed")

    # A function to help change the default_keys if necessary
    def set_default_keys(self, default_keys):
        if isinstance(default_keys, list):
            if all(isinstance(item, str) for item in default_keys):
                self.default_keys = default_keys
                return True

        raise TypeError('Default keys must be given in a list for type<Dict26>')
        # if the function doesnt return true, then we get an error

    def get_default_keys(self):
        return self.default_keys

    # A function to help change the allowed_units if necessary
    def set_allowed_units(self, allowed_units):
        if isinstance(allowed_units, dict):
            if all(isinstance(allowed_units[unit_family], dict) for unit_family in allowed_units):
                if all((isinstance(allowed_units[unit_family][unit], str) for unit in unit_family) for unit_family in
                       allowed_units):
                    self.allowed_units = allowed_units
                    return True

        raise TypeError('Allowed units must be given in a dictionary of a dictionary for type<Dict26>')
        # if the function doesn't return true, then we get an error

    def get_allowed_units(self):
        return self.allowed_units

    # do not use this functions for [...]*int assignments. It breaks the class somehow.
    def initialize_default_keys_to_single_value(self, value):
        for key in self.default_keys:
            self[key] = value
        return True

# after importing the time_units_dict, run this example to see if it works
# a = Dict26(allowed_units={'Time': time_units_dict}, default_keys=['x (us)'], spacer=' ')
# a['x (us)'] = [3, 4, 5]
# b = a['x (ns)']
# print b
