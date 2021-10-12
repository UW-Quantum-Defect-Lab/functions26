# 2021-04-08
# This code was made for use in the Fu lab
# by Vasilis Niaouris


import csv
import json
import numpy as np
import pickle

from dataclasses import dataclass, field
from pandas import DataFrame
from typing import Any, List

from ..useful_functions import add_extension_if_necessary
from ..units.UnitClass import UnitClass
from ..DataFrame26 import DataFrame26
from ..Dict26 import Dict26
from ..useful_functions import is_str_containing_float


@dataclass
class QDLFDataManager:
    data: Any = None
    parameters: dict = field(default_factory=dict)
    datatype: str = None

    def __post_init__(self):

        recognized_data_forms = [DataFrame.__name__, DataFrame26.__name__, dict.__name__, Dict26.__name__,
                                 np.ndarray.__name__, type(None).__name__]
        if type(self.data).__name__ not in recognized_data_forms:
            raise TypeError('QDLFDataManager data must be either a DataFrame, DataFrame26 dict, Dict26,'
                            ' ndarray or NoneType.')

        if type(self.data).__name__ == DataFrame26.__name__:
            try:
                dummy = self.data.default_keys
            except AttributeError:
                self.data.default_keys = self.data.keys()

        if self.parameters is None:
            self.parameters = dict()
        if not isinstance(self.parameters, dict):
            raise TypeError('QDLFDataManager parameters muct be a dictionary.')

        self.set_attributes_from_parameter_dict(self.parameters)

    def __repr__(self):
        string = 'Dataclass QDLFDataManager with attributes:\n'
        string = string + '----------------------------\n'
        for key in self.__dict__.keys():
            if key == 'parameters':
                continue
            string = string + key + ' = '
            if key == 'data':
                string = string + '\n'
            string = string + f'{self.__getattribute__(key)!r}\n'
        string = string + '----------------------------'
        return string

    def set_attributes_from_parameter_dict(self, parameters):
        for key in parameters:
            new_attribute_name = key
            for special_character in ['/', ' ']:
                new_attribute_name = new_attribute_name.replace(special_character, '_')
            self.__setattr__(new_attribute_name.lower(), parameters[key])

    @staticmethod
    def get_stringed_values(key, item):
        if isinstance(item, DataFrame26):
            if item.spacer == '_':
                key = type(item).__name__ + '_ ' + key
            else:
                key = type(item).__name__ + ' ' + key
            item = item.to_dict('list')
        elif isinstance(item, DataFrame):
            key = type(item).__name__ + ' ' + key
            item = item.to_dict('list')
        elif isinstance(item, np.ndarray):
            item = item.tolist()
        elif isinstance(item, UnitClass):
            key = key + ' (' + item.original_unit + ')'
        return key, item

    @staticmethod
    def get_reversed_stringed_values(key, item):
        if type(item) == list:
            item = np.array(item)
        if key.split(' ')[0] == 'DataFrame26':
            item = DataFrame26(data=item, spacer=' ')
            key = key.split(' ')[1]
        elif key.split(' ')[0] == 'DataFrame26_':
            item = DataFrame26(data=item, spacer='_')
            key = key.split(' ')[1]
        elif key.split(' ')[0] == 'DataFrame':
            item = DataFrame(data=item)
            key = key.split(' ')[1]

        if key.split(' ')[-1][0] == '(' and key.split(' ')[-1][-1] == ')':
            item = UnitClass(item, key.split(' ')[-1][1:-1])
            key = key.split(' ')[0]

        return key, item

    def get_json_par_dict(self):
        parameters = dict()
        for key in self.parameters.keys():
            key, item = self.get_stringed_values(key, self.parameters[key])
            parameters[key] = item

        return parameters

    def get_json_dict(self):
        json_dict = {}

        key, item = self.get_stringed_values('data', self.data)
        json_dict[key] = item

        key, item = self.get_stringed_values('datatype', self.datatype)
        json_dict[key] = item

        json_dict['parameters'] = self.get_json_par_dict()

        return json_dict

    @classmethod
    def load(cls, filename) -> "QDLFDataManager":
        if filename.split('.')[-1] == 'qdlf':
            return cls.load_from_qdlf(filename)
        elif filename.split('.')[-1] == 'json':
            return cls.load_from_json(filename)
        elif filename.split('.')[-1] == 'csv':
            return cls.load_from_csv(filename)
        else:
            return cls.load_from_qdlf(filename)

    @classmethod
    def load_from_qdlf(cls, filename) -> "QDLFDataManager":
        filename = add_extension_if_necessary(filename, 'qdlf')

        with open(filename, 'rb') as file:
            obj = pickle.load(file)

        return cls(obj.data, obj.parameters, obj.datatype)

    @classmethod
    def load_from_json(cls, filename) -> "QDLFDataManager":
        filename = add_extension_if_necessary(filename, 'json')
        with open(filename) as file:
            json_dict = json.load(file)
        data = None
        parameters = dict()
        datatype = None
        for key in json_dict.keys():
            item = json_dict[key]
            key, item = cls.get_reversed_stringed_values(key, item)
            if key == 'data':
                data = item
            elif key == 'datatype':
                datatype = item
            elif key == 'parameters':
                for par_key in item.keys():
                    par_item = item[par_key]
                    par_key, par_item = cls.get_reversed_stringed_values(par_key, par_item)
                    parameters[par_key] = par_item

        return cls(data, parameters, datatype)

    @classmethod
    def load_from_csv(cls, filename, header_unit_spacer='_') -> "QDLFDataManager":
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            line_count = 0
            header = None
            data = []
            parameters_csv = dict()
            datatype = None
            for row in reader:
                if line_count == 0:
                    if len(row) > 0:
                        if row[-1] == 'parameters' and row[-2] == 'datatype':
                            data_row = row[:-2]
                        else:
                            data_row = row
                    if np.any([not is_str_containing_float(element) for element in data_row]):
                        header = data_row
                    else:
                        for element in data_row:
                            data.append([element])
                elif line_count == 1:
                    if len(row) > 0:
                        if not (is_str_containing_float(row[-1]) and is_str_containing_float(row[-2])):
                            data_row = row[:-2]
                            if np.all([element == '' for element in data_row]):
                                data_row = []
                            datatype = json.loads(row[-2])
                            parameters_csv = json.loads(row[-1])
                        else:
                            data_row = row
                    if header is None:
                        for i, item in enumerate(data_row):
                            data[i].append(item)
                    else:
                        for element in data_row:
                            data.append([element])
                else:
                    for i, item in enumerate(row):
                        data[i].append(item)
                line_count += 1

        # convert back to orginal dict
        parameters = dict()
        for key in parameters_csv.keys():
            item = parameters_csv[key]
            key, item = cls.get_reversed_stringed_values(key, item)
            parameters[key] = item

        if header is None:
            data = np.array(data, dtype=np.float64)
        else:
            if len(data):
                data = DataFrame26(data={key: data[i] for i, key in enumerate(header)}, spacer=header_unit_spacer)
            else:
                data = DataFrame26(data={key: [] for i, key in enumerate(header)}, spacer=header_unit_spacer)
        return cls(data, parameters, datatype)

    def save(self, filename):
        if filename.split('.')[-1] == 'qdlf':
            self.save_as_qdlf(filename)
        elif filename.split('.')[-1] == 'json':
            self.save_as_json(filename)
        elif filename.split('.')[-1] == 'csv':
            self.save_as_csv(filename)
        else:
            self.save_as_qdlf(filename)

    def save_as_qdlf(self, filename):
        filename = add_extension_if_necessary(filename, 'qdlf')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def save_as_json(self, filename):
        filename = add_extension_if_necessary(filename, 'json')
        json_dict = self.get_json_dict()
        with open(filename, 'w') as file:
            json.dump(json_dict, file)

    def save_as_csv(self, filename, header_unit_spacer='_'):
        filename = add_extension_if_necessary(filename, 'csv')
        if not isinstance(self.data, DataFrame26):
            data_tbs = DataFrame26(data=self.data, spacer=header_unit_spacer)
        else:
            data_tbs = self.data
        rows = [row.split(',') for row in
                data_tbs.to_csv(index=False, header=(not isinstance(self.data, np.ndarray))).split('\r\n')]
        # meaning that if self.data was a np.array, we dont want to save the header
        rows = rows[:-1]  # datadrame.to_csv adds an empty line at the end which we don't want.
        if len(rows) == 1:
            rows.append(['' for i in range(len(rows[0]))])
        if len(rows) == 0:
            rows = [[], []]
        rows[0].append('datatype')
        datatype_string = json.dumps(self.datatype)
        rows[1].append(datatype_string)

        parameter_string = json.dumps(self.get_json_par_dict())
        rows[0].append('parameters')
        rows[1].append(parameter_string)

        with open(filename, 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            for row in rows:
                writer.writerow(row)


class MultiQDLF:

    def __init__(self, qdlf_data_manager_list: List[QDLFDataManager], identifiers: List[str],
                 multidatatype: str = None):

        if len(qdlf_data_manager_list) != len(identifiers):
            raise ValueError('Length of qdlf_data_manager_list must be the same as the one of the identifiers')

        self.data_managers: List[QDLFDataManager] = qdlf_data_manager_list
        self.identifiers = identifiers
        self.multidatatype = multidatatype

    def save(self, filename):
        if filename.split('.')[-1] == 'mqdlf':
            self.save_as_mqdlf(filename)
        elif filename.split('.')[-1] == 'json':
            self.save_as_jsons(filename)
        elif filename.split('.')[-1] == 'csv':
            self.save_as_csvs(filename)
        else:
            self.save_as_mqdlf(filename)

    def save_as_mqdlf(self, filename):
        filename = add_extension_if_necessary(filename, 'mqdlf')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def save_as_jsons(self, filename):
        if filename.split('.')[-1] == 'json':
            filename = ''.join(filename.split('.')[:-1])
        for i, identifier in enumerate(self.identifiers):
            filename = f"{filename}_{self.multidatatype}_{identifier}.json"
            self.data_managers[i].save_as_json(filename)

    def save_as_csvs(self, filename):
        if filename.split('.')[-1] == 'csv':
            filename = ''.join(filename.split('.')[:-1])
        for i, identifier in enumerate(self.identifiers):
            filename_i = f"{filename}_{self.multidatatype}_{identifier}.csv"
            self.data_managers[i].save_as_csv(filename_i)

    @classmethod
    def load(cls, filename):
        filename = add_extension_if_necessary(filename, 'mqdlf')
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return cls(obj.data_managers, obj.identifiers, obj.multidatatype)

