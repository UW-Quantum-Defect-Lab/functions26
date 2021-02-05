# 2020-09-10
# This code was made for use in the Fu lab
# by Christian Zimmermann

import os
import numpy as np
import pandas as pd
import seaborn as sns
# Import constants
from .constants import conversion_factor_nm_to_ev  # eV*nm

# Create dict of dataframes containing PL reference lines:
PL_lines = {}
# ZnO:
# Review: Özgur et al., JAP 98 (2005)
# Free Excitons (FX): Liang et al., PRL 20 (1968)
# Neutral Donor-bound Excitions (D0X):
# low temperature
# Strassburg et al., pss (b) 241 (2004) + Meyer et al., pss (b) 241 (2004) + Kumar et al., # Journal of Luminescence 176 (2016) [extracted from plot using inkscape]
# medium temperature
# Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape]
# TES of D0X:
# InZn, SnZn-LiZn: Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape] + Wagner et al., PRB 84 (2011)
# Hi, GaZn, AlZn: Wagner et al., PRB 84 (2011)
# Ionized Donor-bound Excitons (DpX): Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape]
# Unidentified bound Excitons (I# and Z#): Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape] + Meyer et al., pss (b) 241 (2004)
# Unidentified Excitions bound to structural defects (Y#): Wagner et al., PRB 84 (2011) + Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape]

PL_lines['ZnO'] = pd.DataFrame(
                                data = [
                                ['Free Excitons','FX_A_1',r'FX$_\mathrm{A}^\mathrm{n = 1}$',3.3781,'low','FX<sub>A</sub> (<i>n</i> = 1)'],
                                ['Free Excitons','FX_A_2',r'FX$_\mathrm{A}^\mathrm{n = 2}$',3.4282,'low','FX<sub>A</sub> (<i>n</i> = 2)'],
                                ['Free Excitons','FX_A_3',r'FX$_\mathrm{A}^\mathrm{n = 3}$',3.4375,'low','FX<sub>A</sub> (<i>n</i> = 3)'],
                                ['Free Excitons','FX_B_1',r'FX$_\mathrm{B}^\mathrm{n = 1}$',3.3856,'low','FX<sub>B</sub> (<i>n</i> = 1)'],
                                ['Free Excitons','FX_B_2',r'FX$_\mathrm{B}^\mathrm{n = 2}$',3.4324,'low','FX<sub>B</sub> (<i>n</i> = 2)'],
                                ['Free Excitons','FX_B_3',r'FX$_\mathrm{B}^\mathrm{n = 3}$',3.4412,'low','FX<sub>B</sub> (<i>n</i> = 3)'],
                                ['Free Excitons','FX_C_1',r'FX$_\mathrm{C}^\mathrm{n = 1}$',3.4264,'low','FX<sub>C</sub> (<i>n</i> = 1)'],
                                ['Free Excitons','FX_C_2',r'FX$_\mathrm{C}^\mathrm{n = 2}$',3.4722,'low','FX<sub>C</sub> (<i>n</i> = 2)'],
                                ['Free Excitons','FX_C_3',r'FX$_\mathrm{C}^\mathrm{n = 3}$',3.4808,'low','FX<sub>C</sub> (<i>n</i> = 3)'],
                                ['Neutral Donor-bound Excitons','AlZnX_0',r'Al$_\mathrm{Zn}^\mathrm{0}$-X',3.3608,'low','(Al<sub>Zn</sub>)<sup>0</sup>-X'],
                                ['Neutral Donor-bound Excitons','GaZnX_0',r'Ga$_\mathrm{Zn}^\mathrm{0}$-X',3.3598,'low','(Ga<sub>Zn</sub>)<sup>0</sup>-X'],
                                ['Neutral Donor-bound Excitons','InZnX_0',r'In$_\mathrm{Zn}^\mathrm{0}$-X',3.3567,'low','(In<sub>Zn</sub>)<sup>0</sup>-X'],
                                ['Neutral Donor-bound Excitons','SnZnLiZnX_0',r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{0}$-X',3.3534,
                                                                                'low','(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>0</sup>-X'],
                                ['Neutral Donor-bound Excitons','HiX_0',r'H$_\mathrm{i}^\mathrm{0}$-X',3.3628,'low','(H<sub>i</sub>)<sup>0</sup>-X'],
                                ['I lines','I_0',r'I$_\mathrm{0}$',3.3725,'low','I<sub>0</sub>'],
                                ['I lines','I_1',r'I$_\mathrm{1}$',3.3718,'low','I<sub>1</sub>'],
                                ['I lines','I_1a',r'I$_\mathrm{1a}$',3.3679,'low','I<sub>1a</sub>'],
                                ['I lines','I_2',r'I$_\mathrm{2}$',3.3674,'low','I<sub>2</sub>'],
                                ['I lines','I_3',r'I$_\mathrm{3}$',3.3663,'low','I<sub>3</sub>'],
                                ['I lines','I_5',r'I$_\mathrm{5}$',3.3614,'low','I<sub>5</sub>'],
                                ['I lines','I_6a',r'I$_\mathrm{6a}$',3.3604,'low','I<sub>6a</sub>'],
                                ['I lines','I_7',r'I$_\mathrm{7}$',3.3601,'low','I<sub>7</sub>'],
                                ['I lines','I_7a',r'I$_\mathrm{7a}$',3.3718,'low','I<sub>7a</sub>'],
                                ['I lines','I_8a',r'I$_\mathrm{8a}$',3.3593,'low','I<sub>8a</sub>'],
                                ['I lines','I_11',r'I$_\mathrm{11}$',3.3484,'low','I<sub>11</sub>']
                                ],
                                columns = ['line_class','line_identifier','line_latex','x_eV','temperature', 'line_html']
                                )


class PL_Lines:
    def __init__(self, material = 'ZnO', temperature = 'low'):
        self.material = material
        self.temperature = temperature

        self.import_database()

    def import_database(self):
        self.lines = PL_lines[self.material]
        self.lines = self.lines[self.lines['temperature'] == self.temperature]
        self.lines.reset_index(drop = True, inplace = True)

        self.lines['x_nm'] = conversion_factor_nm_to_ev/self.lines['x_eV']

        return True
