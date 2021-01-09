# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.frequency import *
# compatible with time

# frequency
gigahertz = GHz = 1.  # default units
# bigger
terahertz = THz = 1e-3
petahertz = PHz = 1e-6
exahertz = EHz = 1e-9
# smaller
megahertz = MHz = 1e3
kilohertz = kHz = 1e6
hertz = Hz = 1e9

keys = dir()
frequency_units_dict = {}

for key in keys:
    if key[0] != '_':
        frequency_units_dict[key] = locals().get(key)
