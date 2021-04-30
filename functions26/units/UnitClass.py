# 2020-09-13
# This code was made for use in the Fu lab
# by Vasilis Niaouris
from ..units import unit_families


class UnitClass(float):

    def __new__(cls, value, original_unit=None):
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
        if type(original_unit) != str:
            raise ValueError('Original unit can not be empty')

        self.value = value
        self.original_unit = original_unit
        self.unit_family = None
        for potential_family in unit_families:
            for potential_unit in unit_families[potential_family]:
                if potential_unit == original_unit:
                    self.unit_family = potential_family

        if self.unit_family is None:
            raise ValueError('Unit was not found in available unit families')

        for unit in unit_families[self.unit_family]:
            change_units = unit_families[self.unit_family][unit] / unit_families[self.unit_family][original_unit]
            setattr(self, unit, self.value*change_units)

    def __repr__(self):
        return f'{self.value!r} ' + self.original_unit

    def __add__(self, other) -> "UnitClass":
        if type(other) == UnitClass:
            if self.unit_family == other.unit_family:
                change_units_other = unit_families[other.unit_family][self.original_unit] / \
                                     unit_families[other.unit_family][other.original_unit]
                return UnitClass(self.value + other.value * change_units_other, self.original_unit)
            else:
                raise TypeError('Unit classes are not of the same Unit')
        else:
            return UnitClass(self.value + other, self.original_unit)

    def __radd__(self, other) -> "UnitClass":
        if type(other) == UnitClass:
            if self.unit_family == other.unit_family:
                change_units_self = unit_families[other.unit_family][other.original_unit] / \
                                    unit_families[other.unit_family][self.original_unit]
                return UnitClass(other.value + self.value * change_units_self, other.original_unit)
            else:
                raise TypeError('Unit classes are not of the same Unit')
        else:
            return UnitClass(other + self.value, self.original_unit)

    def __sub__(self, other):
        if type(other) == UnitClass:
            if self.unit_family == other.unit_family:
                change_units_other = unit_families[other.unit_family][self.original_unit] / \
                                     unit_families[other.unit_family][other.original_unit]
                return UnitClass(self.value - other.value * change_units_other, self.original_unit)
            else:
                raise TypeError('Unit classes are not of the same Unit')
        else:
            return UnitClass(self.value - other, self.original_unit)

    def __rsub__(self, other):
        print('In')
        if type(other) == UnitClass:
            if self.unit_family == other.unit_family:
                change_units_self = unit_families[other.unit_family][other.original_unit] / \
                                    unit_families[other.unit_family][self.original_unit]
                return UnitClass(other.value - self.value * change_units_self, other.original_unit)
            else:
                raise TypeError('Unit classes are not of the same Unit')
        else:
            return UnitClass(other - self.value, self.original_unit)

    def __mul__(self, other) -> "UnitClass":
        if isinstance(other, UnitClass):
            return self.value * other.value
        else:
            return UnitClass(self.value * other, self.original_unit)

    def __rmul__(self, other) -> "UnitClass":
        if isinstance(other, UnitClass):
            return self.value * other.value
        else:
            return UnitClass(self.value * other, self.original_unit)

    def __truediv__(self, other) -> "UnitClass":
        if isinstance(other, UnitClass):
            return self.value / other.value
        else:
            return UnitClass(self.value / other, self.original_unit)


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
            raise ValueError('Unit was not found in available unit families')

        for unit in unit_families[self.unit_family]:
            change_units = unit_families[self.unit_family][unit] / unit_families[self.unit_family][original_unit]
            value_list = []
            for v in self.values:
                if v is None:
                    value_list.append(None)
                else:
                    value_list.append(v*change_units)
            setattr(self, unit, value_list)
