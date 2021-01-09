# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.magnetic_field import *

# magnetic field
tesla = T = 1.  # default units
# smaller
millitesla = mT = 1e3
# other units
gauss = Gs = 1e4
milligauss = mGs = 1e7

keys = dir()
magnetic_field_units_dict = {}

for key in keys:
    if key[0] != '_':
        magnetic_field_units_dict[key] = locals().get(key)
