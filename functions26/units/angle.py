# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.angle import *

from scipy.constants import pi


# angle
radians = rad = 1.  # default units
# smaller
milliradians = mrad = 1.e-3
# other units
degrees = deg = 180./pi

keys = dir()
angle_units_dict = {}

for key in keys:
    if key[0] != '_' and key != 'pi':
        angle_units_dict[key] = locals().get(key)
