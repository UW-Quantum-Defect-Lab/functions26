# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.temperature import *

# temperature
kelvin = K = 1.  # default units
# smaller
millikelvin = mK = 1.e3
microkelvin = uK = 1.e6

keys = dir()
temperature_units_dict = {}

for key in keys:
    if key[0] != '_':
        temperature_units_dict[key] = locals().get(key)


def kelvin_to_celsius(temperature_in_kelvin):
    if not isinstance(temperature_in_kelvin, list):
        return temperature_in_kelvin - 273.15
    else:
        return [t-273.15 for t in temperature_in_kelvin]


def celsius_to_fahrenheit(temperature_in_celsius):
    if not isinstance(temperature_in_celsius, list):
        return 1.8*temperature_in_celsius + 32.
    else:
        return [1.8*t + 32. for t in temperature_in_celsius]


def kelvin_to_fahrenheit(temperature_in_kelvin):
    if not isinstance(temperature_in_kelvin, list):
        return 1.8*temperature_in_kelvin - 459.67
    else:
        return [1.8*t - 459.67 for t in temperature_in_kelvin]


def rankine_to_fahrenheit(temperature_in_rankine):
    if not isinstance(temperature_in_rankine, list):
        return temperature_in_rankine + 459.67
    else:
        return [t + 459.67 for t in temperature_in_rankine]


def fahrenheit_to_celsius(temperature_in_fahrenheit):
    if not isinstance(temperature_in_fahrenheit, list):
        return (temperature_in_fahrenheit-32.)/1.8
    else:
        return [(t - 32.)/1.8 for t in temperature_in_fahrenheit]


def fahrenheit_to_kelvin(temperature_in_fahrenheit):
    if not isinstance(temperature_in_fahrenheit, list):
        return (temperature_in_fahrenheit + 459.67)/1.8
    else:
        return [(t + 459.67)/1.8 for t in temperature_in_fahrenheit]


def fahrenheit_to_rankine(temperature_in_fahrenheit):
    if not isinstance(temperature_in_fahrenheit, list):
        return temperature_in_fahrenheit+459.67
    else:
        return [t + 459.67 for t in temperature_in_fahrenheit]
