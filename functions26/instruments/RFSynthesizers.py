# 2021-03-23
# This code was made for use in the Fu lab
# by Vasilis Niaouris

import warnings

from windfreak import SynthHD  # this is a class
from ..InstrumentHandler import Instrument


class WindfreakSythHDInstrument(Instrument):  # it is a serial instrument that has it's own class based on Serial

    def __init__(self, device_name='', verbose=1, initialize_at_definition=True):
        super().__init__(device_name, verbose, initialize_at_definition)

    def initialize_instrument(self):

        try:
            self.instrument = SynthHD(self.device_name)
            self.instrument.init()
            if self.verbose > 1:
                print(self.device_name + ': Instrument successfully initialized')
            return self.instrument
        except Exception as e:
            warnings.warn(self.device_name + ': SynthHD Error. Check instrument name or if instrument is available')
            return -1

    def terminate_instrument(self):

        try:
            self.instrument.close()
        except Exception as e:
            warnings.warn(self.device_name + ': An error occured. Instrument was not terminated. Error: ' + str(e))

    def get_instrument_reading_string(self):
        raise RuntimeError('This read statement is too broad. Check windfreak SynthHD class for more info.')

