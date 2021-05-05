# 2020-09-10
# This code was made for use in the Fu lab
# by Christian Zimmermann
# Last edit: 2021-05-01


import pandas as pd

from .constants import conversion_factor_nm_to_ev  # eV*nm
from .constants import n_air


LO_phonon_energy = {}  # energy of the longitudinal optical phonon
LO_phonon_energy['ZnO'] = 0.072  # eV, Wagner's dissertation

# Create dict of dataframes containing PL reference lines:
PL_lines = {}
# ZnO:
# Review: Özgur et al., JAP 98 (2005)
# Free Excitons (FX): Liang et al., PRL 20 (1968)
# Neutral Donor-bound Excitions (D0X):
# low temperature
# Strassburg et al., pss (b) 241 (2004) + Meyer et al., pss (b) 241 (2004) + Kumar et al., # Journal of Luminescence 176 (2016) [extracted from plot using inkscape]
# TES of D0X:
# InZn, SnZn-LiZn: Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape] + Wagner et al., PRB 84 (2011)
# Hi, GaZn, AlZn: Wagner et al., PRB 84 (2011)
# Ionized Donor-bound Excitons (DpX): Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape]
# Unidentified bound Excitons (I# and Z#): Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape] + Meyer et al., pss (b) 241 (2004)
# Unidentified Excitions bound to structural defects (Y#): Wagner et al., PRB 84 (2011) + Kumar et al., Journal of Luminescence 176 (2016) [extracted from plot using inkscape]

PL_lines['ZnO'] = pd.DataFrame(
    data=[
        ['Free Excitons', 'FX_A_1', r'FX$_\mathrm{A}^\mathrm{n = 1}$', 3.3781, 'low', 'FX<sub>A</sub> (<i>n</i> = 1)'],
        ['Free Excitons', 'FX_A_2', r'FX$_\mathrm{A}^\mathrm{n = 2}$', 3.4282, 'low', 'FX<sub>A</sub> (<i>n</i> = 2)'],
        ['Free Excitons', 'FX_A_3', r'FX$_\mathrm{A}^\mathrm{n = 3}$', 3.4375, 'low', 'FX<sub>A</sub> (<i>n</i> = 3)'],
        ['Free Excitons', 'FX_B_1', r'FX$_\mathrm{B}^\mathrm{n = 1}$', 3.3856, 'low', 'FX<sub>B</sub> (<i>n</i> = 1)'],
        ['Free Excitons', 'FX_B_2', r'FX$_\mathrm{B}^\mathrm{n = 2}$', 3.4324, 'low', 'FX<sub>B</sub> (<i>n</i> = 2)'],
        ['Free Excitons', 'FX_B_3', r'FX$_\mathrm{B}^\mathrm{n = 3}$', 3.4412, 'low', 'FX<sub>B</sub> (<i>n</i> = 3)'],
        ['Free Excitons', 'FX_C_1', r'FX$_\mathrm{C}^\mathrm{n = 1}$', 3.4264, 'low', 'FX<sub>C</sub> (<i>n</i> = 1)'],
        ['Free Excitons', 'FX_C_2', r'FX$_\mathrm{C}^\mathrm{n = 2}$', 3.4722, 'low', 'FX<sub>C</sub> (<i>n</i> = 2)'],
        ['Free Excitons', 'FX_C_3', r'FX$_\mathrm{C}^\mathrm{n = 3}$', 3.4808, 'low', 'FX<sub>C</sub> (<i>n</i> = 3)'],
        ['Neutral Donor-bound Excitons', 'AlZnX_0', r'Al$_\mathrm{Zn}^\mathrm{0}$-X', 3.3608, 'low',
         '(Al<sub>Zn</sub>)<sup>0</sup>-X'],
        ['Neutral Donor-bound Excitons', 'GaZnX_0', r'Ga$_\mathrm{Zn}^\mathrm{0}$-X', 3.3598, 'low',
         '(Ga<sub>Zn</sub>)<sup>0</sup>-X'],
        ['Neutral Donor-bound Excitons', 'InZnX_0', r'In$_\mathrm{Zn}^\mathrm{0}$-X', 3.3567, 'low',
         '(In<sub>Zn</sub>)<sup>0</sup>-X'],
        ['Neutral Donor-bound Excitons', 'SnZnLiZnX_0',
         r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{0}$-X', 3.3534,
         'low', '(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>0</sup>-X'],
        ['Neutral Donor-bound Excitons', 'HiX_0', r'H$_\mathrm{i}^\mathrm{0}$-X', 3.3628, 'low',
         '(H<sub>i</sub>)<sup>0</sup>-X'],
        ['1-Phonon Replica of Neutral Donor-bound Excitons', 'AlZnX_0-LO', r'Al$_\mathrm{Zn}^\mathrm{0}$-X-LO',
         3.3608 - LO_phonon_energy['ZnO'], 'low', '(Al<sub>Zn</sub>)<sup>0</sup>-X-LO'],
        ['1-Phonon Replica of Neutral Donor-bound Excitons', 'GaZnX_0-LO', r'Ga$_\mathrm{Zn}^\mathrm{0}$-X-LO',
         3.3598 - LO_phonon_energy['ZnO'], 'low', '(Ga<sub>Zn</sub>)<sup>0</sup>-X-LO'],
        ['1-Phonon Replica of Neutral Donor-bound Excitons', 'InZnX_0-LO', r'In$_\mathrm{Zn}^\mathrm{0}$-X-LO',
         3.3567 - LO_phonon_energy['ZnO'], 'low', '(In<sub>Zn</sub>)<sup>0</sup>-X-LO'],
        ['1-Phonon Replica of Neutral Donor-bound Excitons', 'SnZnLiZnX_0-LO',
         r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{0}$-X-LO',
         3.3534 - LO_phonon_energy['ZnO'],
         'low', '(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>0</sup>-X-LO'],
        ['1-Phonon Replica of Neutral Donor-bound Excitons', 'HiX_0-LO', r'H$_\mathrm{i}^\mathrm{0}$-X-LO',
         3.3628 - LO_phonon_energy['ZnO'], 'low', '(H<sub>i</sub>)<sup>0</sup>-X-LO'],
        ['2-Phonon Replica of Neutral Donor-bound Excitons', 'AlZnX_0-2LO', r'Al$_\mathrm{Zn}^\mathrm{0}$-X-2LO',
         3.3608 - 2 * LO_phonon_energy['ZnO'], 'low', '(Al<sub>Zn</sub>)<sup>0</sup>-X-2LO'],
        ['2-Phonon Replica of Neutral Donor-bound Excitons', 'GaZnX_0-2LO', r'Ga$_\mathrm{Zn}^\mathrm{0}$-X-2LO',
         3.3598 - 2 * LO_phonon_energy['ZnO'], 'low', '(Ga<sub>Zn</sub>)<sup>0</sup>-X-2LO'],
        ['2-Phonon Replica of Neutral Donor-bound Excitons', 'InZnX_0-2LO', r'In$_\mathrm{Zn}^\mathrm{0}$-X-2LO',
         3.3567 - 2 * LO_phonon_energy['ZnO'], 'low', '(In<sub>Zn</sub>)<sup>0</sup>-X-2LO'],
        ['2-Phonon Replica of Neutral Donor-bound Excitons', 'SnZnLiZnX_0-2LO',
         r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{0}$-X-2LO',
         3.3534 - 2 * LO_phonon_energy['ZnO'],
         'low', '(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>0</sup>-X-2LO'],
        ['2-Phonon Replica of Neutral Donor-bound Excitons', 'HiX_0-2LO', r'H$_\mathrm{i}^\mathrm{0}$-X-2LO',
         3.3628 - 2 * LO_phonon_energy['ZnO'], 'low', '(H<sub>i</sub>)<sup>0</sup>-X-2LO'],
        ['I lines', 'I_0', r'I$_\mathrm{0}$', 3.3725, 'low', 'I<sub>0</sub>'],
        ['I lines', 'I_1', r'I$_\mathrm{1}$', 3.3718, 'low', 'I<sub>1</sub>'],
        ['I lines', 'I_1a', r'I$_\mathrm{1a}$', 3.3679, 'low', 'I<sub>1a</sub>'],
        ['I lines', 'I_2', r'I$_\mathrm{2}$', 3.3674, 'low', 'I<sub>2</sub>'],
        ['I lines', 'I_3', r'I$_\mathrm{3}$', 3.3663, 'low', 'I<sub>3</sub>'],
        ['I lines', 'I_5', r'I$_\mathrm{5}$', 3.3614, 'low', 'I<sub>5</sub>'],
        ['I lines', 'I_6a', r'I$_\mathrm{6a}$', 3.3604, 'low', 'I<sub>6a</sub>'],
        ['I lines', 'I_7', r'I$_\mathrm{7}$', 3.3601, 'low', 'I<sub>7</sub>'],
        ['I lines', 'I_7a', r'I$_\mathrm{7a}$', 3.3718, 'low', 'I<sub>7a</sub>'],
        ['I lines', 'I_8a', r'I$_\mathrm{8a}$', 3.3593, 'low', 'I<sub>8a</sub>'],
        ['I lines', 'I_11', r'I$_\mathrm{11}$', 3.3484, 'low', 'I<sub>11</sub>'],
        ['Z lines', 'Z_1', r'Z$_\mathrm{1}$', 3.3608, 'low', 'Z<sub>1</sub>'],
        ['Z lines', 'Z_2', r'Z$_\mathrm{2}$', 3.3612, 'low', 'Z<sub>2</sub>'],
        ['Z lines', 'Z_3', r'Z$_\mathrm{3}$', 3.3617, 'low', 'Z<sub>3</sub>'],
        ['Z lines', 'Z_4', r'Z$_\mathrm{4}$', 3.3619, 'low', 'Z<sub>4</sub>'],
        ['Ionized Donor-bound Excitons', 'AlZnX_p', r'Al$_\mathrm{Zn}^\mathrm{+}$-X', 3.3734, 'low',
         '(Al<sub>Zn</sub>)<sup>+</sup>-X'],
        ['Ionized Donor-bound Excitons', 'GaZnX_p', r'Ga$_\mathrm{Zn}^\mathrm{+}$-X', 3.3718, 'low',
         '(Ga<sub>Zn</sub>)<sup>+</sup>-X'],
        ['Ionized Donor-bound Excitons', 'InZnX_p', r'In$_\mathrm{Zn}^\mathrm{+}$-X', 3.3676, 'low',
         '(In<sub>Zn</sub>)<sup>+</sup>-X'],
        ['Ionized Donor-bound Excitons', 'SnZnLiZnX_p',
         r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{+}$-X', 3.3632,
         'low', '(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>+</sup>-X'],
        ['Y lines', 'Y_1', r'Y$_\mathrm{1}$', 3.3328, 'low', 'Y<sub>1</sub>'],
        ['Y lines', 'Y_2', r'Y$_\mathrm{2}$', 3.3363, 'low', 'Y<sub>2</sub>'],
        ['Y lines', 'Y_3', r'Y$_\mathrm{3}$', 3.3465, 'low', 'Y<sub>3</sub>'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2s)', 'TES_2s_AlZnX_0',
         r'Al$_\mathrm{Zn}^\mathrm{0}$-X (TES, 2s)', 3.3228, 'low', '(Al<sub>Zn</sub>)<sup>0</sup>-X (TES, 2s)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2s)', 'TES_2s_GaZnX_0',
         r'Ga$_\mathrm{Zn}^\mathrm{0}$-X (TES, 2s)', 3.3191, 'low', '(Ga<sub>Zn</sub>)<sup>0</sup>-X (TES, 2s)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2s)', 'TES_2s_InZnX_0',
         r'Ga$_\mathrm{Zn}^\mathrm{0}$-X (TES, 2s)', 3.3101, 'low', '(In<sub>Zn</sub>)<sup>0</sup>-X (TES, 2s)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2s)', 'TES_2s_HiX_0',
         r'H$_\mathrm{i}^\mathrm{0}$-X (TES, 2s)', 3.3278, 'low', '(H<sub>i</sub>)<sup>0</sup>-X (TES, 2s)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2s)', 'TES_2s_SnZnLiZnX_0',
         r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{0}$-X (TES, 2s)', 3.2986,
         'low', '(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>0</sup>-X (TES, 2s)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2p)', 'TES_2p_AlZnX_0',
         r'Al$_\mathrm{Zn}^\mathrm{0}$-X (TES, 2p)', 3.3220, 'low', '(Al<sub>Zn</sub>)<sup>0</sup>-X (TES, 2p)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2p)', 'TES_2p_GaZnX_0',
         r'Ga$_\mathrm{Zn}^\mathrm{0}$-X (TES, 2p)', 3.3177, 'low', '(Ga<sub>Zn</sub>)<sup>0</sup>-X (TES, 2p)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2p)', 'TES_2p_InZnX_0',
         r'In$_\mathrm{Zn}^\mathrm{0}$-X (TES, 2p)', 3.3061, 'low', '(In<sub>Zn</sub>)<sup>0</sup>-X (TES, 2p)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2p)', 'TES_2p_HiX_0',
         r'H$_\mathrm{i}^\mathrm{0}$-X (TES, 2p)', 3.3287, 'low', '(H<sub>i</sub>)<sup>0</sup>-X (TES, 2p)'],
        ['Two-electron satellites of Neutral Donor-bound Excitons (2p)', 'TES_2p_SnZnLiZnX_0',
         r'$\left(\mathrm{Sn}_\mathrm{Zn}-\mathrm{Li}_\mathrm{Zn}\right)^\mathrm{0}$-X (TES, 2p)', 3.2929,
         'low', '(Sn<sub>Zn</sub>-Li<sub>Zn</sub>)<sup>0</sup>-X (TES, 2p)']
    ],
    columns=['line_class', 'line_identifier', 'line_identifier_latex', 'x_eV', 'temperature', 'line_identfier_html']
)


class PL_Lines:
    def __init__(self, material='ZnO', temperature='low', correct_for_nair=False):
        self.material = material
        self.temperature = temperature
        self.correct_for_nair = correct_for_nair

        self._import_database()

    def _import_database(self):
        self.database = PL_lines[self.material]
        self.database = self.database[self.database['temperature'] == self.temperature]
        self.database.reset_index(drop=True, inplace=True)
        self.database['x_nm'] = conversion_factor_nm_to_ev / self.database['x_eV']

        if self.correct_for_nair:
            self.database['x_eV'] = conversion_factor_nm_to_ev / (self.database['x_nm'] * n_air)

        return True
