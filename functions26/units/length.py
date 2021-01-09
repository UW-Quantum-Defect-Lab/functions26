# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.length import *

# length
micrometers = um = 1.  # default units
# bigger
millimeters = mm = 1.e-3
centimeters = cm = 1.e-4
meters = 1.e-6
kilometers = km = 1.e-9
# smaller
nanometers = nm = 1.e3
picometers = pm = 1.e6
femtometers = fm = fermi = 1.e9
angstroms = 1.e4
# other units
inches = 2.54*meters
# smaller
thou = th = 25.4*um
# bigger
feet = 12*inches
yards = 3*feet
miles = 5280*feet

keys = dir()
length_units_dict = {}

for key in keys:
    if key[0] != '_':
        length_units_dict[key] = locals().get(key)
