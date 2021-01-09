# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.energy import *

# energy
electronvolts = eV = 1.  # default units
# bigger
kiloelectronvolts = keV = 1.e-3
megaelectronvolts = MeV = 1.e-6
gigaelectronvolts = GeV = 1.e-9
teraelectronvolts = TeV = 1.e-12
# smaller
millielectronvolts = meV = 1.e3
microelectronvolts = ueV = 1.e6
nanoelectronvolts = neV = 1.e9
# other units
joules = 1.602176565e-19*eV
# bigger
kilojoules = 1.e-3*joules
megajoules = 1.e-6*joules
gigajoules = 1.e-9*joules
# smaller
millijoules = 1.e3*joules
microjoules = 1.e6*joules
nanojoules = 1.e9*joules

keys = dir()
energy_units_dict = {}

for key in keys:
    if key[0] != '_':
        energy_units_dict[key] = locals().get(key)
