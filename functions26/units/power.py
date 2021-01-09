# Added/Edited by Chris on 2020-09-21
# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.power import *

# power
microwatts = uW = 1e-3  # default units
# bigger
milliwatts = mW = 1e-6
watts = W = 1e-9
kilowatts = kW = 1.e-12
megawatts = MW = 1.e-15
# smaller
nanowatts = nW = 1

keys = dir()
power_units_dict = {}

for key in keys:
    if key[0] != '_':
        power_units_dict[key] = locals().get(key)
