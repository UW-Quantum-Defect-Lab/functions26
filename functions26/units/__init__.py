# Example of how this works.
# Example 1 (straightforward)
# You have a variable in microseconds named delay_time.
# You want to use them in milliseconds, so you just call it as delay_time*ms
# Example 2 (backwards)
# You import a file and the variable is not in the default unit (e.g. delay_time is in nanoseconds)
# You do delay_time = delay_time/ns -> this should take it to the default unit (microseconds)
# Example 3 (complicated)
# You import variable delay_time in nanoseconds and you want seconds
# You do delay_time = delay_time/ns*seconds

from .angle import angle_units_dict
from .area import area_units_dict
from .energy import energy_units_dict
from .frequency import frequency_units_dict
from .length import length_units_dict
from .magnetic_field import magnetic_field_units_dict
from .power import power_units_dict
from .temperature import temperature_units_dict
from .time import time_units_dict
from .volume import volume_units_dict

unit_families = {'Angle': angle_units_dict,
                 'Area': area_units_dict,
                 'Energy': energy_units_dict,
                 'Frequency': frequency_units_dict,
                 'Length': length_units_dict,
                 'Magnetic Field': magnetic_field_units_dict,
                 'Power': power_units_dict,
                 'Temperature': temperature_units_dict,
                 'Time': time_units_dict,
                 'Volume': volume_units_dict}
