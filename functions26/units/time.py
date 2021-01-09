# 2020-09-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris
# Call me as: from units.time import *
# compatible with frequency

# time
nanoseconds = ns = 1.   # default units
# bigger
microseconds = us = 1.e-3
milliseconds = ms = 1e-6
seconds = secs = 1e-9
minutes = mins = secs/60.
hours = mins/60.
days = hours/24.
weeks = days/7.
years = days/365.
mean_years = days/365.2425
# smaller
picoseconds = ps = 1e3
femtoseconds = fs = 1e6

keys = dir()
time_units_dict = {}

for key in keys:
    if key[0] != '_':
        time_units_dict[key] = locals().get(key)
