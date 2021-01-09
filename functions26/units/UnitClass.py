# 2020-09-13
# This code was made for use in the Fu lab
# by Vasilis Niaouris
from ..units.__init__ import unit_families


class UnitClass(float):

    def __new__(cls, value, original_unit):
        try:
            if value is None:
                return None
            else:
                return float.__new__(cls, value)
        except TypeError:
            raise TypeError
        except ValueError:
            raise ValueError

    def __init__(self, value, original_unit):

        self.value = value
        self.original_unit = original_unit
        self.unit_family = None
        for potential_family in unit_families:
            for potential_unit in unit_families[potential_family]:
                if potential_unit == original_unit:
                    self.unit_family = potential_family

        if self.unit_family is None:
            raise RuntimeError('Unit was not found in available unit families')

        for unit in unit_families[self.unit_family]:
            change_units = unit_families[self.unit_family][unit] / unit_families[self.unit_family][original_unit]
            setattr(self, unit, self.value*change_units)


class UnitClassList(list):

    # def __new__(cls, values, original_unit):
    #     try:
    #         if values is None:
    #             return None
    #         else:
    #             return list.__new__(cls, values)
    #     except TypeError:
    #         raise TypeError
    #     except ValueError:
    #         raise ValueError

    def __init__(self, values, original_unit):
        super(UnitClassList, self).__init__(values)
        self.values = values
        self.original_unit = original_unit
        self.unit_family = None
        for potential_family in unit_families:
            for potential_unit in unit_families[potential_family]:
                if potential_unit == original_unit:
                    self.unit_family = potential_family

        if self.unit_family is None:
            raise RuntimeError('Unit was not found in available unit families')

        for unit in unit_families[self.unit_family]:
            change_units = unit_families[self.unit_family][unit] / unit_families[self.unit_family][original_unit]
            value_list = []
            for v in self.values:
                if v is None:
                    value_list.append(None)
                else:
                    value_list.append(v*change_units)
            setattr(self, unit, value_list)
