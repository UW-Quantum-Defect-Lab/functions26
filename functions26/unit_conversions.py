from scipy.constants import c, h, hbar
from . import units
from .units.UnitClass import UnitClass


def wavelength_to_freq(wavelength, units_in='nm', units_out='GHz'):
    if wavelength <= 0:
        raise ValueError('Wavelength can not be any value below zero')
    wl = UnitClass(wavelength, units_in)
    sol = UnitClass(c, 'meters')
    try:
        unit_conv = units.unit_families['Frequency'][units_out]/units.unit_families['Frequency']['GHz']
    except KeyError:
        raise ValueError(units_out + ' not found in unit_family Frequency ')
    freq = UnitClass(sol/wl.nm*unit_conv, units_out)

    return freq


def wavelength_to_frequency_difference(w_i, w_f, units_in='nm', units_out='GHz'):
    fr_f = wavelength_to_freq(w_i, units_in, units_out)
    fr_i = wavelength_to_freq(w_f, units_in, units_out)
    df = UnitClass(fr_f - fr_i, units_out)

    return df


def wavelength_to_frequency_bandwidth(w_c, w_b, units_c='nm', units_b='nm', units_out='GHz'):
    # w_b is the total width
    w_i = UnitClass(UnitClass(w_c, units_c) - UnitClass(w_b, units_b).__getattribute__(units_c)/2., units_c)
    w_f = UnitClass(UnitClass(w_c, units_c) + UnitClass(w_b, units_b).__getattribute__(units_c)/2., units_c)
    fr_f = wavelength_to_freq(w_i, units_c, units_out)
    fr_i = wavelength_to_freq(w_f, units_c, units_out)
    df = UnitClass(fr_f - fr_i, units_out)

    return df
