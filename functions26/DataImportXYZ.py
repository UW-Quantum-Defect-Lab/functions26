# 2020-09-26
# This code was made for use in the Fu lab
# by Christian Zimmermann

import pandas as pd


class DataImportXYZ:
    def __init__(self, file_name):
        self.file_name = file_name

        self.data = self.get_data()

    def get_data(self):
        data = pd.read_csv(self.file_name, delimiter = ' ', header = 0)
        return data


class DataImportMagnetLogger(DataImportXYZ):
    def __init__(self, file_name):
        super().__init__(file_name)

    def get_data(self):
        data = pd.read_csv(self.file_name, delim_whitespace = True)
        data.dropna(how='all', axis=1, inplace = True)
        data.columns = ['Time', 'Pressure (psi)', 'Fill Status', 'Cold-Head Temperature (K)',
                        'Sample Temperature (K)', 'Cold-Head Heater Output (%)', 'Helium Level (inches)']
        data.Time = pd.to_datetime(data.Time)

        return data
