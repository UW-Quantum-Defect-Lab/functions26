# Last modified: 2022-02-24
# by Vasilis Niaouris

from collections import UserDict
import warnings

from typing import Dict, Union, Tuple, List


def get_unit(key: str, allowed_units: Dict[str, Dict[str, Union[float, int]]],
             spacer: str) -> Tuple[Union[str, None], Union[str, None]]:
    """
    The function get_unit receives a Dict26 key (e.g. "pumpOnTime_us" or "Frequency (MHz)") and a dictionary
    of unit families where each family contains a dictionary of units (e.g. {'Time': {'ns': 1., 'us': 1.e-3},
    'Frequency': {'GHz': 1., 'MHz': 1.e3}}). The functions check if the last part of the key (the unit)
     belongs in any of the unit families. It then returns the key's unit and the corresponding unit family.

    Parameters
    ----------
    key: str
        A key from a dict26 class (e.g. "pumpOnTime_us" or "Frequency (MHz)").
    allowed_units: Dict[str, Dict[str, Union[float, int]]]
        A dictionary of unit families where each family contains a dictionary of units
        (e.g. {'Time': {'ns': 1., 'us': 1.e-3}, 'Frequency': {'GHz': 1., 'MHz': 1.e3}}).
    spacer: str
        The spacer of the dict26 class (e.g. "pumpOnTime_us" -> "_", "Frequency (MHz)" -> " ").

    Returns
    -------
    Union[str, None], Union[str, None]
        A tuple with two strings. The first string is the units included in the provided key, and the second is the unit
        family in which they belong to. If the unit is part of the allowed unit families, the function returns
        None values.

    Examples
    --------
    >>> allowed_units = {'Length': {'nm': 1., 'um': 1.e-3}, 'Energy': {'eV': 1., 'meV': 1.e3}}
    >>> print(get_unit('x_nm', allowed_units, '_'))

    This would return ('nm', 'Length').
    """

    key_unit = get_unit_string(key, spacer)
    for unit_family in allowed_units:
        for unit in allowed_units[unit_family]:
            if unit == key_unit:
                return key_unit, unit_family

    return None, None


def get_dictionary_key(key: str, dictionary_keys: List, allowed_units: Dict[str, Dict[str, Union[float, int]]],
                       unit_family: str, spacer: str):
    """
    The function get_dictionary_key checks if the key name without the units of a Dict26 key is part of the dictionary.
    It returns the initial part of the given key and the original dictionary key.

    Parameters
    ----------
    key: str
        A key from a dict26 class (e.g. "pumpOnTime_us" or "Frequency (MHz)").
    dictionary_keys: List
        A list of all the keys in the dictionary of interest.
    allowed_units: Dict[str, Dict[str, Union[float, int]]]
        A dictionary of unit families where each family contains a dictionary of units
        (e.g. {'Time': {'ns': 1., 'us': 1.e-3}, 'Frequency': {'GHz': 1., 'MHz': 1.e3}}).
    unit_family: str
        The unit family the key's units belong to (e.g. for "nm" it would be "Length").
    spacer: str
        The spacer of the dict26 class (e.g. "pumpOnTime_us" -> "_", "Frequency (MHz)" -> " ").

    Returns
    -------
    Union[str, None], Union[str, None]
    A tuple with two strings. The first string is the base key included in the provided key, and the second is the
    key with its original units. If the base key in not part of the list of keys or the unit is not part of the
    allowed unit families, the function returns None values.
    """
    key_initial = get_init_string(key, spacer)
    for dict_key in dictionary_keys:
        if key_initial == get_init_string(dict_key, spacer) \
                and get_unit_string(dict_key, spacer) in allowed_units[unit_family]:
            return key_initial, dict_key

    return None, None


def get_init_string(key, spacer):
    """
    The function get_init_string allows the user to provide a Dict26 key (e.g. "pumpOnTime_us" or "Frequency (MHz)")
    and the corresponding spacer (e.g. "_" or " ") and returns the string of BEFORE the unit
    within the key (e.g. "pumpOnTime" or "Frequency").

    Parameters
    ----------
    key: str
        A key from a dict26 class (e.g. "pumpOnTime_us" or "Frequency (MHz)")
    spacer: str
        The spacer of the dict26 class (e.g. "pumpOnTime_us" -> "_", "Frequency (MHz)" -> " ")

    Returns
    -------
    str:
        the string of the name within the given Dict26 key (e.g. "pumpOnTime" or "Frequency").
    """
    string_list = key.split(spacer)[:-1]
    return ' '.join(string_list)


def get_unit_string(key: str, spacer: str):
    """
    The function get_unit_string allows the user to provide a Dict26 key (e.g. "pumpOnTime_us" or "Frequency (MHz)")
    and the corresponding spacer (e.g. "_" or " ") and returns the string of the unit
    within the key (e.g. "us" or "MHz").

    Parameters
    ----------
    key: str
        A key from a dict26 class (e.g. "pumpOnTime_us" or "Frequency (MHz)")
    spacer: str
        The spacer of the dict26 class (e.g. "pumpOnTime_us" -> "_", "Frequency (MHz)" -> " ")

    Returns
    -------
    str:
        the string of the unit within the given Dict26 key (e.g. "us" or "MHz").
    """
    string_list = key.split(spacer)[-1]  # split is redundant for '_' style, but important for ' ' spacer
    return string_list.strip('()')  # the [0] is because it will return a list with a single string in it


class Dict26(UserDict):
    """
    Dict26 is a UserDict that allows the user to change the units of the contained data to all the different possible
    units. It is able to restring the values to only a default set of values, although this is kind of an overkill and
    most likely a mistake. Nevertheless, we are rolling with it for now.

    Attributes
    ----------
    default_keys: list, default=[]
        A list of the default keys of the dictionary.
    allowed_units: Dict[str, Dict[str, Union[float, int]]], default={}
        A dictionary of allowed units families for the Dict26 keys.
    spacer: str, default='_'
        The string used to separate between the base part of a key and its unit.
    restrict_to_defaults: bool, default=False
        A parameter to determine if new keys are allowed to be set or not.

    Examples
    --------
    >>> allowed_units = {'Length': {'nm': 1., 'um': 1.e-3}, 'Energy': {'eV': 1., 'meV': 1.e3}}
    >>> default_keys = ['x_nm', 'x_eV']
    >>> a = Dict26(allowed_units, default_keys, spacer='_')
    >>> a['x_nm'] = [3, 4, 5]
    >>> b = a['x_um']

    b should contain a list of [0.003, 0.004, 0.005].
    """

    def __init__(self, default_keys: List = None, allowed_units: Dict[str, Dict[str, Union[float, int]]] = None,
                 spacer: str = None, restrict_to_defaults: bool = False, *args, **kwargs):
        """
        Parameters
        ----------
        default_keys: list, default=[]
            A list of the default keys of the dictionary.
        allowed_units: Dict[str, Dict[str, Union[float, int]]], default={}
            A dictionary of allowed units families for the Dict26 keys.
        spacer: str, default='_'
            The string used to separate between the base part of a key and its unit.
        restrict_to_defaults: bool, default=False
            A parameter to determine if new keys are allowed to be set or not.
        """
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
            print('DataFrame26 spacer was not given. Set to default: _')
            self.spacer = '_'
        else:
            warnings.warn('DataFrame26 spacer not valid. Set to default: _')
            self.spacer = '_'
        self.restrict_to_defaults = restrict_to_defaults

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """
        This function is called when one is assigning a new value to a - potentially new, but not necessarily - key
        (e.g. somedict26object[some_key] = some_value).
        We overload setitem, to make sure we are not adding any keys not listed in default_keys.
        """
        if self.restrict_to_defaults and key not in self.default_keys and key not in self.keys():
            # The third if-statement is here in case you change default_keys and the old keys are not in them.
            raise RuntimeError("Appending keys not listed in default_keys of type<Dict26> is not allowed")
        else:
            super(Dict26, self).__setitem__(key, value)

    def __getitem__(self, key):
        """
        This function is called when one is trying to access the preexisting value of a given key
        (e.g. some_variable = somedict26object[some_key]).
        We overload getitem to check if the requested key exists either in the key list, or it is an allowed
        unit variant (e.g. if "Frequency (MHz)" is a key, then "Frequency (GHz)" can also be called successfully).
        """
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

    def __delitem__(self, key):
        """
        We overload delitem to avoid deleting keys.
        """
        raise RuntimeError("Deletion in type<Dict26> not allowed")

    def pop(self, s=None):
        """
        We overload pop to avoid deleting keys.
        """
        raise RuntimeError("Deletion/Popping in type<Dict26> not allowed")

    def popitem(self, s=None):
        """
        We overload popitem to avoid deleting keys.
        """
        raise RuntimeError("Deletion/Popping in type<Dict26> not allowed")

    def set_default_keys(self, default_keys: List):
        """
        A function to override the default_keys if necessary

        Parameters
        ----------
        default_keys: List
        """
        if isinstance(default_keys, list):
            if all(isinstance(item, str) for item in default_keys) or len(default_keys) == 0:
                # after or we check for empty lists
                self.default_keys = default_keys
                return True

        raise TypeError('Default keys must be given in a list for type<Dict26>')

    def get_default_keys(self) -> List:
        """
        Returns
        -------
        List
            The default keys.
        """
        return self.default_keys

    def set_allowed_units(self, allowed_units: Dict[str, Dict[str, Union[float, int]]]):
        """
        A function to override the allowed_units if necessary.

        Parameters
        ----------
        allowed_units: Dict[str, Dict[str, Union[float, int]]]
        """
        if isinstance(allowed_units, dict):
            if all(isinstance(allowed_units[unit_family], dict) for unit_family in allowed_units):
                if all((isinstance(allowed_units[unit_family][unit], str) for unit in unit_family) for unit_family in
                       allowed_units):
                    self.allowed_units = allowed_units
                    return True

        raise TypeError('Allowed units must be given in a dictionary of a dictionary for type<Dict26>')

    def get_allowed_units(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Returns
        -------
        Dict[str, Dict[str, Union[float, int]]]
            The allowed units.
        """
        return self.allowed_units

    def initialize_default_keys_to_single_value(self, value):
        """
        A function to initialize all the default keys. It initializes them to a single value (e.g. 4, 'apples', None).
        Unfortunately, using this to assign vaules of the type [0]*10 or [None]*10, does not work right now.

        Parameters
        ----------
        value:
            Any value of your choice to initialize every default key.
        """
        for key in self.default_keys:
            self[key] = value
        return True
