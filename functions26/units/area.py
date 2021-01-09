# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.area import *

from .length import *

# area
micrometers2 = um2 = micrometers**2  # default units
# bigger
millimeters2 = mm2 = millimeters**2
centimeter2s = cm2 = centimeters**2
meter2 = meters**2
kilometesr2 = km2 = kilometers**2
# smaller
nanometesr2 = nm2 = nanometers**2
picometers2 = pm2 = picometers**2
femtometers2 = fm2 = fermi2 = femtometers**2
angstroms2 = angstroms**2
# other units
inches2 = inches**2
# smaller
thou2 = th2 = thou**2
# bigger
feet2 = feet**2
yards2 = yards**2
miles2 = miles**2

keys = dir()
area_units_dict = {}

for key in keys:
    if key[0] != '_' and key[-1] == '2':
        area_units_dict[key] = locals().get(key)
