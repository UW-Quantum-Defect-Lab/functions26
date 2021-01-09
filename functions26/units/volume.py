# 2020-09-12
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.volume import *

from .length import *

# volume
micrometers3 = um3 = micrometers**3  # default units
# bigger
millimeters3 = mms3 = millimeters**3
centimeter3 = cm3 = centimeters**3
meters3 = meters**3
kilometers3 = km3 = kilometers**3
# smaller
nanometers3 = nm3 = nanometers**3
picometers3 = pm3 = picometers**3
femtometers3 = fm3 = fermi3 = femtometers**3
angstroms3 = angstroms**3
# other units
inches3 = inches**3
# smaller
thou3 = th3 = thou**3
# bigger
feet3 = feet**3
yards3 = yards**3
miles3 = miles**3

keys = dir()
volume_units_dict = {}

for key in keys:
    if key[0] != '_' and key[-1] == '3':
        volume_units_dict[key] = locals().get(key)
