# 2020-09-10
# This code was made for use in the Fu lab
# by Vasilis Niaouris
import numpy as np
import pandas as pd
import io, pkgutil

from scipy.constants import e, h, c

c_nm = c*1e+9  # nm/s
h_eV = h/e  # eV*s
conversion_factor_nm_to_ev = h_eV*c_nm  # eV*nm
n_air = 1.000293
n_air_737p8 = 1.0002754800150584
n_air_368p9 = 1.0002846825653557


packaged_data = pkgutil.get_data('functions26.external_data', 'refractive_index_air.csv')
n_air_database = pd.read_csv(io.BytesIO(packaged_data))


def get_n_air(value, value_units='nm air'):
    if value_units not in ['nm air', 'nm vacuum', 'eV', 'THz']:
        raise ValueError('value_units must be in [nm air, nm vacuum, eV, THz]')

    value_column = ''
    if value_units == 'nm air':
        value_column = 'Wavelength Air (um)'
        value = value/1000
    elif value_units == 'nm vacuum':
        value_column = 'Wavelength Vacuum (um)'
        value = value/1000
    elif value_units == 'eV':
        value_column = 'Energy (eV)'
    elif value_units == 'THz':
        value_column = 'Frequency (THz)'

    return np.interp(value, n_air_database[value_column], n_air_database['Refractive Index'])
