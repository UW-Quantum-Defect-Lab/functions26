import numpy as np

from scipy.constants import c, h, hbar
from . import units
from .units.UnitClass import UnitClass


def wavelength_to_freq(wavelength, units_in='nm', units_out='GHz'):

    islist = isinstance(wavelength, list) or isinstance(wavelength, np.ndarray)
    if not islist:
        wavelength = [wavelength]

    sol = UnitClass(c, 'meters') # speed of light
    try:
        unit_conv = units.unit_families['Frequency'][units_out] / units.unit_families['Frequency']['GHz']
    except KeyError:
        raise ValueError(units_out + ' not found in unit_family Frequency ')

    freq = []
    for wl in wavelength:
        if wl <= 0:
            raise ValueError('Wavelength can not be any value below zero')
        wl = UnitClass(wl, units_in)
        freq.append(UnitClass(sol/wl.nm*unit_conv, units_out))

    freq = np.array(freq, dtype=UnitClass)
    if islist:
        return freq
    else:
        return freq[0]


def wavelength_to_frequency_difference(w_i, w_f, units_in='nm', units_out='GHz'):
    fr_f = wavelength_to_freq(w_i, units_in, units_out)
    fr_i = wavelength_to_freq(w_f, units_in, units_out)
    df = fr_f - fr_i

    return df


def wavelength_to_frequency_bandwidth(w_c, w_b, units_c='nm', units_b='nm', units_out='GHz'):

    wc_islist = isinstance(w_c, list) or isinstance(w_c, np.ndarray)
    wb_islist = isinstance(w_b, list) or isinstance(w_b, np.ndarray)

    if not wc_islist and wb_islist:
        len_wb = len(w_b)
        w_c = [w_c for i in range(len_wb)]
        len_wc = len_wb
    elif wc_islist and not wb_islist:
        len_wc = len(w_c)
        w_b = [w_b for i in range(len_wc)]
        len_wb = len_wc
    elif not wc_islist and not wb_islist:
        w_c = [w_c]
        w_b = [w_b]
        len_wc = len_wb = 1
    else:
        len_wc = len(w_c)
        len_wb = len(w_b)

    if len_wc == len_wb:
        w_i = [UnitClass(w_c[i], units_c) - UnitClass(w_b[i], units_b) / 2 for i in range(len_wc)]
        w_f = [UnitClass(w_c[i], units_c) + UnitClass(w_b[i], units_b) / 2 for i in range(len_wc)]
    else:
        raise ValueError('The length size of the list w_c and w_b do not much')

    df = wavelength_to_frequency_difference(w_i, w_f, units_c, units_out)

    if not wb_islist and not wc_islist:
        return df[0]
    else:
        return df

