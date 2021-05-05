# 2019-05-12 / last updated on 2020-09-15
# This code was made for use in the Fu lab
# by Vasilis Niaouris

import numpy as np
import nidaqmx
import pandas as pd
import pyvisa
import serial as srl
import time
import warnings


class Instrument:

    def __init__(self, device_name, verbose=1, initialize_at_definition=True):
        self.device_name = device_name
        self.verbose = verbose
        self.initialize_at_definition = initialize_at_definition

        if self.initialize_at_definition and not self.device_name == '':
            self.instrument = self.initialize_instrument()
        else:
            self.instrument = None

    def initialize_instrument(self):
        warnings.warn('Assign your own initialize_instrument()')
        pass

    def terminate_instrument(self):
        warnings.warn('Assign your own terminate_instrument()')
        pass

    def reopening_session(self):
        if self.verbose > 1:
            print(self.device_name + ': Reopening session')
        self.instrument = self.initialize_instrument()

    def get_instrument_reading_string(self):
        warnings.warn('Assign your own get_instrument_reading_string()')
        pass

    def simple_write(self, write_command):
        try:
            return self.instrument.write(write_command)
        except AttributeError:
            raise AttributeError(str(type(self).__name__) + ' does not support a write command.')

    def simple_read(self):
        return self.instrument.read()

    def simple_query(self, write_command):
        try:
            self.instrument.write(write_command)
            return self.instrument.read()
        except AttributeError:
            raise AttributeError(str(type(self).__name__) + ' does not support a write command.')


class GPIBInstrument(Instrument):
    read_command_number = 0

    def __init__(self, device_name='', read_termination='', read_command='', time_delay=1e-17, clear_command='*CLS',
                 verbose=1, initialize_at_definition=True):

        self.read_termination = read_termination
        self.time_delay = time_delay
        self.clear_command = clear_command
        self.read_command = read_command
        if isinstance(self.read_command, str):
            self.read_command_number = 1
        else:
            self.read_command_number = len(read_command)

        self.rm = pyvisa.ResourceManager()  # Resource manager
        super().__init__(device_name, verbose, initialize_at_definition)

    def initialize_instrument(self):

        try:
            self.instrument = self.rm.open_resource(self.device_name)
            self.instrument.write(self.clear_command)
            self.instrument.read_termination = self.read_termination
            if self.verbose > 1:
                print(self.device_name + ': Instrument successfully initialized')
            return self.instrument
        except pyvisa.errors.VisaIOError:
            warnings.warn(self.device_name + ': Vis IO Error occurred. Check instrument name or '
                                             'if instrument is available')
            return -1

    def terminate_instrument(self):

        try:
            self.instrument.last_status
        except pyvisa.errors.InvalidSession:
            warnings.warn(self.device_name + ': Invalid Session. Instrument might already be closed')
            return -1
        except AttributeError:
            warnings.warn(self.device_name + ': Invalid Session. Instrument might be used by other Program')
            return -1

        try:
            # self.instrument.clear()
            self.instrument.write(self.clear_command)
            self.instrument.close()
            self.instrument = None
            if self.verbose > 1:
                print(self.device_name + ': Instrument successfully terminated')
            return 0
        except not pyvisa.errors.InvalidSession:
            warnings.warn(self.device_name + ': Instrument not terminated')
            return 1

    def get_instrument_reading_string_all(self):
        results_list = []
        for i in range(self.read_command_number):
            self.instrument.write(self.read_command[i])
            time.sleep(self.time_delay)
            results_list.append(self.instrument.read())
        return results_list

    def get_instrument_reading_string(self, read_command=None):
        if read_command is None:
            if self.read_command_number <= 1:
                read_command = self.read_command
            else:
                read_command = self.read_command[0]
                print(
                    self.device_name + ': This device has more than one reading commands. I chose to read the first one.')
        self.instrument.write(read_command)
        time.sleep(self.time_delay)
        return self.instrument.read()


class NIdaqInstrument(Instrument):
    available_channel_types = ['ai', 'ctr']
    available_channel_reading_types = {'ai': ['Voltage'], 'ctr': ['EdgeCount']}
    available_channel_lengths = {'ai': 2, 'ctr': 3}

    def __init__(self, device_name='', verbose=1, initialize_at_definition=True, channel_type=None,
                 channel_number_list=None,
                 channel_reading_type=None, response_function_method=None, response_function_file=None,
                 response_function_poly_order=4):
        self.task = None
        self.channels = None
        self.channel_name_length = None

        # getting channel type
        if channel_type is None:  # Assume channel name is in device name, i.e.: 'dev1/ai0' -> 'ai'
            self.channel_type = device_name.split('/')[-1][:2]
            if self.channel_type == 'ct':
                self.channel_type = 'ctr'
        else:
            self.channel_type = channel_type

        if self.channel_type not in self.available_channel_types:
            warnings.warn('Channel type \'' + self.channel_type + '\' is not yet supported. Visit '
                                                                  'https://nidaqmx-python.readthedocs.io/en/latest/channel_collection.html to find out how to '
                                                                  'add your own.You can access the created task via:'
                                                                  '<instance_name>.task or <instance_name>.instrument')

        if channel_reading_type is None:
            self.channel_reading_type = self.available_channel_reading_types[self.channel_type][
                0]  # defaulting to voltage supported by analog input channels
            warnings.warn('Channel reading type set to default value \'' +
                          self.available_channel_reading_types[self.channel_type][0] + '\'')
        else:
            self.channel_reading_type = channel_reading_type

        if self.channel_reading_type not in self.available_channel_reading_types[self.channel_type]:
            warnings.warn('Channel reading type \'' + self.channel_reading_type + '\' is not yet supported. Visit '
                                                                                  'https://nidaqmx-python.readthedocs.io/en/latest/channel_collection.html to find out how to '
                                                                                  'add your own. You can access the created task via:'
                                                                                  '<instance_name>.task or <instance_name>.instrument')
        # getting channel numbers
        if channel_number_list is None:  # Assume channel number is in device name, i.e.: 'dev1/ai02' -> '02' -> ['0', '2']
            self.channel_number_list = [number for number in device_name.split('/')[-1][len(self.channel_type):]]
        elif isinstance(channel_number_list, list):
            self.channel_number_list = [str(number) for number in channel_number_list]
        else:
            self.channel_number_list = [str(self.channel_number_list)]

        #  getting device name to create a list of strings with all channel lists that will later be initialized
        if channel_type is None:
            device_name_temp = '/'.join(
                device_name.split('/')[:-1])  # getting only the device name without the channels
        else:
            device_name_temp = device_name
        self.channel_list = [device_name_temp + '/' + self.channel_type + number for number in self.channel_number_list]

        # giving a response function method overwrites the classes default method, resulting in dismissal of
        # the response fucntion file
        if response_function_method is not None:
            self.response_function_method = response_function_method

        # the response function file must be a csv file with two columns, one row for variable names.
        # the first column must be the output you want, the second must be the voltage on the daq.
        self.response_function_file = response_function_file
        self.response_function_poly_order = response_function_poly_order

        self.response_curve = [1, 0]  # This is the default linear curve for no changes in output
        if self.response_function_file is not None:
            file_df = pd.read_csv(self.response_function_file)
            self.response_curve = np.polyfit(file_df[file_df.keys()[1]], file_df[file_df.keys()[0]],
                                             self.response_function_poly_order)

        super().__init__(device_name, verbose, initialize_at_definition)

    def initialize_instrument(self):

        try:
            self.task = nidaqmx.Task()  # NI DAQmx task
            if self.channel_type == 'ai':
                if self.channel_reading_type == 'Voltage':
                    self.channels = [self.task.ai_channels.add_ai_voltage_chan(channel) for channel
                                     in self.channel_list]
            elif self.channel_type == 'ctr':
                if self.channel_reading_type == 'EdgeCount':
                    self.channels = [self.task.ci_channels.add_ci_count_edges_chan(channel) for channel
                                     in self.channel_list]
            if self.verbose > 1:
                print(self.device_name + ': Instrument successfully initialized')
            return self.task
        except nidaqmx.errors.Error:
            warnings.warn(self.device_name + ': NI DAQmx Method Error. Check instrument name or '
                                             'if instrument is available')
            return -1

    def terminate_instrument(self):

        try:
            if self.task.is_task_done():
                try:
                    self.task.close()  # same as self.instrument.close()
                    self.task = None
                    self.channel = None
                    if self.verbose > 1:
                        print(self.device_name + ': Instrument successfully terminated')
                except nidaqmx.errors.Error:
                    warnings.warn(self.device_name + ': NI DAQmx Method Error. Could not clear task.')
            else:
                warnings.warn(self.device_name + ': Task is still running')
        except nidaqmx.errors.DaqError:
            if self.verbose > 1:
                print(self.device_name + ': Task already terminated')

    def get_instrument_reading_string(self):
        reading = self.task.read()  # for NIdaq, the reading is a floating number (voltage)
        return self.response_function(reading)

    def response_function(self, reading):
        return np.polyval(self.response_curve, reading)


class SerialInstrument(Instrument):
    read_command_number = 0

    def __init__(self, device_name='', read_command='', time_out=1, baud_rate=9600,
                 verbose=1, initialize_at_definition=True):

        self.time_out = time_out
        self.baud_rate = baud_rate
        self.read_command = read_command
        if not isinstance(self.read_command, list):
            self.read_command_number = 1
        else:
            self.read_command_number = len(read_command)

        super().__init__(device_name, verbose, initialize_at_definition)

    def initialize_instrument(self):

        try:
            self.instrument = srl.Serial(self.device_name, self.baud_rate, timeout=self.time_out)
            self.instrument.reset_input_buffer()
            self.instrument.reset_output_buffer()
            if self.verbose > 1:
                print(self.device_name + ': Instrument successfully initialized')
            return self.instrument
        except ValueError:
            warnings.warn(self.device_name + ': Serial Port Error. A parameter is out of range or access was denied.')
            return -1
        except srl.SerialException:
            warnings.warn(
                self.device_name + ': Serial Port Error. Instrument can not be found or can not be configured.')
            return -2

    def terminate_instrument(self):

        if self.instrument.is_open:
            try:
                self.instrument.reset_input_buffer()
                self.instrument.reset_output_buffer()
                self.instrument.close()
                self.instrument = None
                if self.verbose > 1:
                    print(self.device_name + ': Instrument successfully terminated')
                return 0
            except Exception as e:
                warnings.warn(self.device_name + ': An error occured. Instrument was not terminated. Error: ' + str(e))
                return -1
        else:
            warnings.warn(self.device_name + ': Instrument might already be closed')
            return -1

    def get_instrument_reading_string_all(self):
        results_list = []
        for i in range(self.read_command_number):
            self.instrument.write(self.read_command[i])
            results_list.append(self.instrument.readlines()[-1])
        return results_list

    def get_instrument_reading_string(self, read_command=None):
        if read_command is None:
            if self.read_command_number <= 1:
                read_command = self.read_command
            else:
                read_command = self.read_command[0]
                print(
                    self.device_name + ': This device was given more than one reading commands. Will only read the first one.')
        self.instrument.write(read_command)
        lines = self.instrument.readlines()
        return lines[-1]

