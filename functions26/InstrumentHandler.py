# 2019-05-12 / last updated on 2020-09-15
# This code was made for use in the Fu lab
# by Vasilis Niaouris
from typing import Union, Any

import numpy as np
import nidaqmx
import pandas as pd
import pyvisa
import serial as srl
import time
import warnings


class Instrument:
    """
    Instrument is an umbrella class which helps the user define different types of instruments under the same structure,
    and eases the user's experience on simple tasks such as initialization, querying and termination.
    With this class, it is easy to attach and detach your device, without holding it "hostage" against other
    programs. Instrument is a class that was written to be used as a superclass and does not function independently.

    Attributes
    ----------
    device_name: str
        The device input port or characteristic ID.
    verbose: int
        The higher the number, the more text for troubleshooting you get.
    initialize_at_definition: bool
        Attempts to initialize the instrument when the object is defined.
    instrument: Any
        The object of the instrument that will be defined in the subclass.
    """

    def __init__(self, device_name: str, verbose: int = 1, initialize_at_definition: bool = True):
        """
        Parameters
        ----------
        device_name: str
            The device input port or characteristic ID.
        verbose: int
            The higher the number, the more text for troubleshooting you get.
        initialize_at_definition: bool
            Attempts to initialize the instrument when the object is defined.
        """
        self.device_name = device_name
        self.verbose = verbose
        self.initialize_at_definition = initialize_at_definition

        if self.initialize_at_definition and not self.device_name == '':
            self.instrument = self.initialize_instrument()
        else:
            self.instrument = None

    def initialize_instrument(self):
        """A sequence that will be used to initialize the connection with the give device."""
        warnings.warn('Assign your own initialize_instrument()')
        pass

    def terminate_instrument(self):
        """A sequence that will be used to terminate the connection with the give device."""
        warnings.warn('Assign your own terminate_instrument()')
        pass

    def reopening_session(self):
        """ A function that simply re-initializes the device. """
        if self.verbose > 1:
            print(self.device_name + ': Reopening session')
        self.instrument = self.initialize_instrument()

    def get_instrument_reading_string(self):
        """When initializing the device, we can set the Instrument class to hold a specific command that the user wants
         to read. This command is useful in the situations where the given device is used primarily to read out a type
         of single values."""
        warnings.warn('Assign your own get_instrument_reading_string()')
        pass

    def simple_write(self, write_command: str):
        """
        To use for simple writing operations.

        Parameters
        ----------
        write_command: str
            String that is fed to the device.
        """
        try:
            return self.instrument.write(write_command)
        except AttributeError:
            raise AttributeError(str(type(self).__name__) + ' does not support a write command.')

    def simple_read(self) -> Union[str, float, int, Any]:
        """
        To use for simple reading operations.

        Returns
        -------
        Union[str, float, int, Any]
            The device output. Nominally a string. Maybe float or int. If user defined, it can be of any type.
        """
        return self.instrument.read()

    def simple_query(self, write_command: str):
        """
        To use for simple querying operations.

        Parameters
        ----------
        write_command: str
            String that is fed to the device.

        Returns
        -------
        Union[str, float, int, Any]
            The device output. Nominally a string. Maybe float or int. If user defined, it can be of any type.
        """
        try:
            self.instrument.write(write_command)
            return self.instrument.read()
        except AttributeError:
            raise AttributeError(str(type(self).__name__) + ' does not support a write command.')


class GPIBInstrument(Instrument):
    """
    GPIBInstrument is a subclass of Instrument. We use the pyvisa library to incorporate some basic functionalities.

    Attributes
    ----------
    device_name: str
        The device input port or characteristic ID.
    verbose: int
        The higher the number, the more text for troubleshooting you get.
    initialize_at_definition: bool
        Attempts to initialize the instrument when the object is defined.
    instrument: Any
        The object of the instrument that will be defined in the subclass.
    read_termination: str
        The characters that terminate each output line of the device (e.g. '\n')
    clear_command: str
        The command to clear the output of the device (e.g. '*CLS')
    time_delay: float
        The time required between consecutive reads. It has to do with how often the device can send it's output to the
        computer. Do NOT use 0.
    read_command: Union[str, List[str]]
        A single string or a list of strings of commands that will be regularly used.
        Admittedly, this is not greatly implemented. Hold your breath for an updated version.
    read_command_number: int
        The length of the read_command list.
    rm: pyvisa.ResourceManager
        The pyvisa resource manager object, through which we communicate with the device.
    """
    read_command_number = 0

    def __init__(self, device_name: str = '', read_termination: str = '', read_command: str = '',
                 time_delay: float = 1e-17, clear_command: str = '*CLS', verbose: int = 1,
                 initialize_at_definition: bool = True):
        """
        Parameters
        ----------
        device_name: str
            The device input port or characteristic ID.
        read_termination: str
            The characters that terminate each output line of the device (e.g. '\n')
        read_command: str
            A single string or a list of strings of commands that will be regularly used.
        time_delay: float
            The time required between consecutive reads. It has to do with how often the device can send it's output to
            the computer. Do NOT use 0.
        clear_command: str
            The command to clear the output of the device (e.g. '*CLS')
        verbose: int
            The higher the number, the more text for troubleshooting you get.
        initialize_at_definition: bool
            Attempts to initialize the instrument when the object is defined.
        """

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
        """
        Returns
        -------
        Union[pyvisa.Resource, int]
            Returns either the pyvisa.Resource or -1 in case of initialization issues.
        """
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
        """
        Returns
        -------
        int
            Returns 0 if the termination was successful, otherwise returns 1.
        """
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
        """
        Uses the user-defined list of reading strings given during the initialization and queries the device, taking
        into account the defined time delay. The query parses through the list by list index number.

        Returns
        -------
        List[str]
            Returns the list of output strings of the queried strings.
        """

        results_list = []
        for i in range(self.read_command_number):
            self.instrument.write(self.read_command[i])
            time.sleep(self.time_delay)
            results_list.append(self.instrument.read())
        return results_list

    def get_instrument_reading_string(self, read_command: Union[None, str] = None):
        """
        Uses a user-defined reading string (defaults to the one given at initialization) and queries the device,
        taking into account the defined time delay.

        Parameters
        ----------
        read_command: Union[None, str]
            A string to query the device. Defaults to the one given at initialization.
            Must be a single string and NOT a list!
        Returns
        -------
        str
            Returns the output of the queried string.
        """
        if read_command is None:
            if self.read_command_number <= 1:
                read_command = self.read_command
            else:
                read_command = self.read_command[0]
                print(
                    self.device_name + ': This device has more than one reading commands. I chose to read the first '
                                       'one.')
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
            warnings.warn('Channel type \'' + self.channel_type +
                          '\' is not yet supported. Visit '
                          'https://nidaqmx-python.readthedocs.io/en/latest/channel_collection.html '
                          'to find out how to add your own.You can access the created task via: <instance_name>.task '
                          'or <instance_name>.instrument')

        if channel_reading_type is None:
            self.channel_reading_type = self.available_channel_reading_types[self.channel_type][
                0]  # defaulting to voltage supported by analog input channels
            warnings.warn('Channel reading type set to default value \'' +
                          self.available_channel_reading_types[self.channel_type][0] + '\'')
        else:
            self.channel_reading_type = channel_reading_type

        if self.channel_reading_type not in self.available_channel_reading_types[self.channel_type]:
            warnings.warn('Channel reading type \'' + self.channel_reading_type +
                          '\' is not yet supported. Visit '
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
            self.task.stop()
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
                warnings.warn(self.device_name + ': An error occurred. Instrument was not terminated. Error: ' + str(e))
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
                print(self.device_name +
                      ': This device was given more than one reading commands. Will only read the first one.')
        self.instrument.write(read_command)
        lines = self.instrument.readlines()
        return lines[-1]
