# 2021-03-23 / last updated on 2021-04-08
# This code was made for use in the Fu lab
# by Vasilis Niaouris


import numpy as np
import pandas as pd

from ..InstrumentHandler import GPIBInstrument
from ..units.frequency import frequency_units_dict
from ..units.UnitClass import UnitClass


class DataGenerator2020AInstrument(GPIBInstrument):
    block_define_cmd = 'DATA:BLOCK:DEFINE'
    memory_cmd = 'DATA:MSIZe'
    mode_state_cmd = 'MODE:STATE'
    oscillator_internal_frequency_cmd = 'SOURCE:OSCILLATOR:INTERNAL:FREQUENCY'
    oscillator_internal_plllock_cmd = 'SOURCE:OSCILLATOR:INTERNAL:PLLlock'
    oscillator_source_cmd = 'SOURCE:OSCILLATOR:SOURCE'
    pattern_cmd = 'DATA:PATTern:BIT'
    running_cmd = 'RUNNING'
    sequence_add_cmd = 'DATA:SEQUENCE:ADD'
    sequence_define_cmd = 'DATA:SEQUENCE:DEFINE'
    subsequence_define_cmd = 'DATA:SUBSEQUENCE:DEFINE'
    start_cmd = 'STARt'
    stop_cmd = 'STOP'

    # Hardware limited options
    allowed_mode_states = ['REPEAT', 'SINGLE', 'STEP', 'ENHANCED']
    allowed_oscillator_sources = ['INTERNAL', 'EXTERNAL']
    allowed_oscillator_frequency_input_units = ['HZ', 'KHZ', 'MHZ']
    oscillator_internal_frequency_limits_in_hz = [1e-2, 2e8]
    bit_position_limits = [0, 35]
    memory_length_limits = [1, 2**16]
    start_address_limits = [0, 2**16 - 1]
    sequence_repetition_limits = [1, 2**16]

    # for all reads except DATA:PATTERN:BIT, the output of the datagenerator of a query command '[CMD]?' is:
    # output_string = ':[CMD] [OUTPUT]\n'
    # Hence, to convert the response to a number or bool, we take the substring: output_string[len([CMD])+2:-1]

    def __init__(self, device_name='', memory=None, oscillator_internal_frequency_in_hz=None, timeout=20000, verbose=1,
                 initialize_at_definition=True):

        # initializing the class as part of the daq superclass (which is part of the instrument super class)
        super().__init__(device_name, verbose=verbose, initialize_at_definition=initialize_at_definition)

        # set the operation timeout in microseconds of the data generator. If the generator takes longer than the
        # timeout, it will throw an error.
        self.instrument.timeout = timeout

        # initializing memory and oscillator internal frequency. They can be kept the same as before if left blank,
        # and changed later if needed. We make sure to change them after the DG stops.
        self.was_running_on_initialization = self.is_running()
        if initialize_at_definition and not self.was_running_on_initialization:
            if oscillator_internal_frequency_in_hz is not None:
                self.set_oscillator_internal_frequency(oscillator_internal_frequency_in_hz, units='Hz')
            if memory is not None:
                self.set_memory(memory)

        # if it was running on initialization (True), the above if will not be completed, hence we need to set
        # the parameters in the future (True)
        self.need_to_set_initiazation_parameters = self.was_running_on_initialization
        if self.need_to_set_initiazation_parameters:
            self.stored_initialization_frequency_hz = oscillator_internal_frequency_in_hz
            self.stored_initialization_memory_size = memory
        # self.simple_write(self.clear_command)

    @staticmethod
    def get_cmd_response_string(read_line, cmd):
        return read_line[len(cmd) + 2:-1]

    @staticmethod
    def pattern_array_to_string(pattern_array):
        # Takes an numpy.ndarray or list, converts it to string, the split erases the empty spaces and seperates
        # 0s and 1s, join brings them all together again, and [1:-1] because the string is of the form '[01101]'
        # np.array2string(pattern_array, , precision=0, separator='', suppress_small=True)
        # return ''.join(str(pattern_array).split())[1:-1]
        pat_str = [str(element) for element in pattern_array]
        return ''.join(pat_str)

    @staticmethod
    def pattern_string_to_array(pattern_string):
        # take the character and convert it to int for all characters in the pattern string
        return np.array([int(character) for character in pattern_string])

    @staticmethod
    def join_pattern_arrays(pattern_list):
        if not (isinstance(pattern_list, np.ndarray) or isinstance(pattern_list, list)):
            raise TypeError('Pattern_list in not numpy.ndarray or list type')
        if not (np.all([isinstance(pattern, np.ndarray) for pattern in pattern_list]) or
                np.all([isinstance(pattern, list) for pattern in pattern_list])):
            raise TypeError('One or more patterns in pattern_list are of different type AND/OR'
                            ' not numpy.ndarray or list type.')

        # take an empty numpy.ndarray and append each pattern given in the pattern list, with index priority
        # (1st pattern is first in the final pattern, 2nd pattern is second in the final pattern etc.)
        final_pattern = np.array(np.concatenate(pattern_list, axis=0))

        return final_pattern

    @staticmethod
    def repeat_pattern_array(pattern_array, repetitions):
        if not isinstance(pattern_array, np.ndarray) or not isinstance(pattern_array, list):
            raise TypeError('Pattern_array in not numpy.ndarray or list type')

        # similar to join_pattern_arrays, takes a zero numpy.ndarray & adds to it the given pattern 'repetitions' times
        final_pattern_array = np.array([])
        for j in range(repetitions):
            final_pattern_array = np.append(final_pattern_array, pattern_array)

        return final_pattern_array

    def generate_OFF_pattern_array(self, length=None):
        # returns an array of zeros, of the specified length
        if length is None:
            length = self.get_memory()

        pattern = np.zeros(length, dtype=int)
        return pattern

    def generate_ON_pattern_array(self, length=None):
        # returns an array of ones, of the specified length
        if length is None:
            length = self.get_memory()

        pattern = np.ones(length, dtype=int)
        return pattern

    def generate_OFF_ON_pattern_array(self, size=1, length=None):
        # returns an array of zeros-ones, of the specified length. Step size is determined from size
        if length is None:
            length = self.get_memory()

        pattern = np.array([int(np.ceil((i + 1 + size)/size) % 2) for i in range(length)])
        return pattern

    def generate_ON_OFF_pattern_array(self, size=1, length=None):
        # returns an array of ones-zeros, of the specified length. Step size is determined from size
        if length is None:
            length = self.get_memory()

        pattern = np.array([int(np.ceil((i + 1)/size) % 2) for i in range(length)])
        return pattern

    def generate_complex_pattern_array(self, dataframe):
        # The first key 'patterns' of the dictionary is the pattern arrays in nd.array or str
        # The second key 'repetitions' is the times the pattern of the same index is repeated
        # The values on the first index go first, then the ones on the second index, etc

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError('Must be given pandas.DataFrame type.')
        if 'patterns' not in dataframe.keys() or 'repetitions' not in dataframe.keys():
            raise ValueError("One or more required keys were not found.\n"
                             "Required keys: 1. \'patterns\' and 2. \'repetitions\'.")

        pattern_list = []
        for i, pattern in enumerate(dataframe['patterns']):
            if isinstance(pattern, str):
                pattern = self.pattern_string_to_array(pattern)
            elif not isinstance(pattern, np.ndarray):
                raise ValueError('Pattern on dataframe index {:.0f} was neither a string nor a numpy.ndarray.')
            pattern_list.append(self.repeat_pattern_array(pattern, dataframe['repetitions'][i]))

        final_pattern = self.join_pattern_arrays(pattern_list)
        return final_pattern

    def define_block(self, names, starting_positions):
        if isinstance(names, str):
            names = [names]
        if not np.all([isinstance(name, str) for name in names]):
            raise TypeError('Data generator block names must be a string or a list of strings.')

        if not np.all([len(names) <= 8 for name in names]):
            raise TypeError('Data generator block names must be a string or a list of strings.')
        if isinstance(starting_positions, int):
            starting_positions = [starting_positions]
        if not np.all([isinstance(sp, int) for sp in starting_positions]):
            raise TypeError('Data generator block starting positions must be a integer or a list of integers.')
        if not np.all(np.diff(starting_positions)) > 0:
            raise ValueError('Data generator block starting positions must be different from each other and increase'
                             ' in value')
        if len(names) != len(starting_positions):
            raise ValueError('Data generator block names and start_positions list lengths need to be equal.')

        # convert numbers to strings
        starting_positions = [str(r) for r in starting_positions]

        # construct a string of the type [block1start_pos],[block1name]\n[block2start_pos],[block2name]...
        block_definition_string = '\n'.join([starting_positions[i] + ',' + names[i].upper() for i in range(len(names))])

        definition_length = str(len(block_definition_string))
        definition_length_digits = str(len(definition_length))

        command = '\n' + self.block_define_cmd + ' #' + definition_length_digits + definition_length +\
                  block_definition_string + '\n' + '\n'

        return self.simple_write(command)

    def set_memory(self, memory):
        if isinstance(memory, int):
            memory = str(memory)
        else:
            raise TypeError('Data generator memory must be an integer.')
        return self.simple_write(self.memory_cmd + ' ' + memory)

    def get_memory(self):
        self.simple_write(self.memory_cmd + '?')
        read_line = self.simple_read()
        value = self.get_cmd_response_string(read_line, self.memory_cmd)
        return int(value)

    def set_mode_state(self, mode_state):
        if mode_state.upper() not in self.allowed_mode_states:
            raise ValueError('Data generator mode state can only be one of: ' + str(self.allowed_mode_states))
        return self.simple_write(self.mode_state_cmd + ' ' + mode_state.upper())

    def get_mode_state(self):
        read_line = self.simple_write(self.mode_state_cmd + '?')
        value = self.get_cmd_response_string(read_line, self.mode_state_cmd)
        return value

    def set_oscillator_internal_frequency(self, freq, units='MHz'):
        if not (isinstance(freq, float) or isinstance(freq, int)):
            raise TypeError('Data generator oscillator internal frequency must be an integer or a float.')

        if units.upper() not in self.allowed_oscillator_frequency_input_units:
            raise ValueError('Data Generator internal frequency input units can only be Hz, kHz, or MHz.')

        # print(UnitClass(freq, units))
        if not (self.oscillator_internal_frequency_limits_in_hz[0] <= UnitClass(freq, units).Hz <=
                self.oscillator_internal_frequency_limits_in_hz[1]):
            raise ValueError('Data Generator internal frequency input should be between 1e-2 and 2e8 Hz')

        freq = "{:.1f}".format(freq)
        return self.simple_write(self.oscillator_internal_frequency_cmd + ' ' + freq + units.upper())

    def get_oscillator_internal_frequency(self, units='MHz'):
        if units not in frequency_units_dict.keys():
            raise ValueError('Data generator oscillator internal frequency units requested were not recognized.')
        self.simple_write(self.oscillator_internal_frequency_cmd + '?')
        read_line = self.simple_read()
        value = self.get_cmd_response_string(read_line, self.oscillator_internal_frequency_cmd)
        return UnitClass(float(value)/frequency_units_dict['Hz']*frequency_units_dict[units], units)

    def set_oscillator_internal_plllock(self, value):
        if isinstance(value, bool):
            value = str(int(value))
        else:
            raise TypeError('Data generator oscillator internal ppllock value must be a boolean.')
        return self.simple_write(self.oscillator_internal_plllock_cmd + ' ' + value)

    def get_oscillator_internal_plllock(self):
        self.simple_write(self.oscillator_internal_plllock_cmd + '?')
        read_line = self.simple_read()
        value = self.get_cmd_response_string(read_line, self.oscillator_internal_plllock_cmd)
        return bool(int(value))

    def set_oscillator_source(self, oscillator_source):
        if oscillator_source.upper() not in self.allowed_oscillator_sources:
            raise ValueError('Data generator mode state can only be one of: ' + str(self.allowed_oscillator_sources))
        return self.simple_write(self.oscillator_source_cmd + ' ' + oscillator_source.upper())

    def get_oscillator_source(self):
        self.simple_write(self.oscillator_source_cmd + '?')
        read_line = self.simple_read()
        value = self.get_cmd_response_string(read_line, self.oscillator_source_cmd)
        return value

    def set_pattern(self, bit_position, pattern_array, start_address=0, length=None):
        # you can set a list/numpy.ndarray of bit_positions with the same pattern at once
        if isinstance(bit_position, int):
            bit_position = [bit_position]
        for bp in bit_position:
            if not self.bit_position_limits[0] <= bp <= self.bit_position_limits[1]:
                raise ValueError('Data generator pattern bit position needs to be between 0 and 35.')

        if not isinstance(pattern_array, np.ndarray):
            raise TypeError('Data generator pattern needs to be numpy.ndarray type')

        if np.any(pattern_array < 0) or np.any(pattern_array > 1):
            raise ValueError('Data generator pattern elements need to be integer 0 or 1')

        if not self.start_address_limits[0] <= start_address <= self.start_address_limits[1]:
            raise ValueError('Data generator pattern starting address needs to be between 0 and 65535.')

        if length is None:
            length = pattern_array.size
        elif length != pattern_array.size:
            raise ValueError('Data generator pattern array length must be the same as the given pattern length.')

        if not self.memory_length_limits[0] <= length <= self.memory_length_limits[1]:
            raise ValueError('Data generator pattern length needs to be between 1 and 65536.')

        # convert numbers to strings
        bit_position = ["{:.0f}".format(bp) for bp in bit_position]
        start_address = "{:.0f}".format(start_address)
        length = "{:.0f}".format(length)
        digits = "{:.0f}".format(len(length))  # length is a string, so its len() gives the number of digits as an int.
        pattern_string = self.pattern_array_to_string(pattern_array)

        command = [self.pattern_cmd + ' ' + bp + ',' + start_address + ',' + length +
                   ',#' + digits + length + pattern_string for bp in bit_position]

        command = '\n' + '\n'.join(command) + '\n'

        return self.instrument.write(command)

        # return [self.simple_write(cmd) for cmd in command] does not work
        # the write function does not like to be saved somewhere :/

    def get_pattern(self, bit_position, start_address=0, length=None):
        if not self.bit_position_limits[0] <= bit_position <= self.bit_position_limits[1]:  # hardware limits
            raise ValueError('Data generator pattern bit position needs to be between 0 and 35.')
        if not self.start_address_limits[0] <= start_address <= self.start_address_limits[1]:  # hardware limits
            raise ValueError('Data generator pattern starting address needs to be between 0 and 65535.')
        if length is None:
            length = self.get_memory()
        elif not self.memory_length_limits[0] <= length <= self.memory_length_limits[1]:  # hardware limits
            raise ValueError('Data generator pattern length needs to be between 1 and 65536.')

        # convert numbers to strings
        bit_position = "{:.0f}".format(bit_position)
        start_address = "{:.0f}".format(start_address)
        length = "{:.0f}".format(length)

        command = self.pattern_cmd + '? ' + bit_position + ',' + start_address + ',' + length
        self.simple_write(command)
        read_line = self.simple_read()

        # The output is of the form ':[CMD] [BITPOS],[STARTADD],[LEN] #[DIGITSOFPATTERN][LEN][PATTERN]\n'
        # Since we know the length of the pattern we requested, we can take ony the last digits we care about
        # (minus the new line)
        pattern_string = read_line[-int(length)-1:-1]  # we take only the last pattern length digits
        return self.pattern_string_to_array(pattern_string)

    def is_running(self):
        self.simple_write(self.running_cmd + '?')
        read_line = self.simple_read()
        value = self.get_cmd_response_string(read_line, self.running_cmd)
        return bool(int(value))

    def add_sequence_step(self, name, line_number=0, repetitions=1, jump_to_line_number=0, wait_on_trigger=0,
                          event_jump=0, infinite_loop=1):
        # sequences only matter if the mode state is enhanced
        # when you want to use a sequence of blocks that have the same attributes defined below, you should define a
        # subsequence with all the blocks you want to use, and then put the subsequence under the sequence.
        # different blocks of patterns can be repeated different amounts of times. same for subsequences and sequences

        # name is the name of the subsequence or the block you want to add to the sequence
        # Line number: the line number of the new sequence. Usually, we just overwrite line_number 0 for all of our
        #              experiments
        # Repetitions: The amount of times the sequence will be repeated before stopping, if the mode state is NOT
        #              'ENHANCED', or not sequence is not an infinite loop
        # Jump to line number: The sequence line number to which the system will jump in case of an event.
        # Wait on trigger: Waiting on a trigger (??? not sure what this does)
        # Event_jump: Bool in case we jump or not in an event. If yes, the sequence line will change to the
        #             'jump_to_line_number' line
        # Infinite loop: If True, the sequence will run indefinately.
        #                If false, the sequence will repeat itself 'repetitions' times before stopping (I think)
        if not isinstance(name, str):
            raise TypeError('Data generator subsequence/block name to be appended in the sequence should be a string.')
        if not self.sequence_repetition_limits[0] <= repetitions <= self.sequence_repetition_limits[1]:
            raise ValueError('Data generator sequence step repetitions should be between 1 and 65536.')
        if wait_on_trigger != 0 and wait_on_trigger != 1:
            raise ValueError('Data generator sequence step wait_one_trigger should be either 0 or 1.')
        if event_jump != 0 and event_jump != 1:
            raise ValueError('Data generator sequence step event_jump should be either 0 or 1.')
        if infinite_loop != 0 and infinite_loop != 1:
            raise ValueError('Data generator sequence step infinite_loop should be either 0 or 1.')

        # convert name to right format
        name = "\"" + name + "\""

        # convert numbers to strings
        line_number = "{:.0f}".format(line_number)
        repetitions = "{:.0f}".format(repetitions)
        jump_to_line_number = "{:.0f}".format(jump_to_line_number)
        wait_on_trigger = "{:.0f}".format(wait_on_trigger)
        event_jump = "{:.0f}".format(event_jump)
        infinite_loop = "{:.0f}".format(infinite_loop)

        command = self.sequence_add_cmd + ' ' + line_number + ',' + name.upper() + ',' + repetitions + ',' + \
                  jump_to_line_number + ',' + wait_on_trigger + ',' + event_jump + ',' + infinite_loop

        return self.simple_write(command)

    def define_sequence(self, names, repetitions=None, jump_to_line_number=None, wait_on_trigger=None, event_jump=None,
                        infinite_loop=None):
        # sequences only matter if the mode state is enhanced
        # when you want to use a sequence of blocks that have the same attributes defined below, you should define a
        # subsequence with all the blocks you want to use, and then put the subsequence under the sequence.
        # different blocks of patterns can be repeated different amounts of times. same for subsequences and sequences

        # Line number: the line number of the new sequence. Usually, we just overwrite line_number 0 for all of our
        #              experiments
        # Repetitions: The amount of times the sequence will be repeated before stopping, if the mode state is NOT
        #              'ENHANCED', or not sequence is not an infinite loop
        # Jump to line number: The sequence line number to which the system will jump in case of an event.
        # Wait on trigger: Waiting on a trigger (??? not sure what this does)
        # Event_jump: Bool in case we jump or not in an event. If yes, the sequence line will change to the
        #             'jump_to_line_number' line
        # Infinite loop: If True, the sequence will run indefinately.
        #                If false, the sequence will repeat itself 'repetitions' times before stopping (I think)
        if isinstance(names, str):
            names = [names]
        if not np.all([isinstance(n, str) for n in names]):
            raise TypeError('Data generator subsequence/block_names to be appended in sequence step must be a string or'
                            ' a list of strings.')

        if repetitions is None:
            repetitions = [1]*len(names)
        if not np.all([self.sequence_repetition_limits[0] <= rep <= self.sequence_repetition_limits[1]
                       for rep in repetitions]):
            raise ValueError('One or more data generator subsequence/block repetitions are not between 1 and 65536.')

        if jump_to_line_number is None:
            jump_to_line_number = [0]*len(names)
        if not np.all([0 <= jtln <= 65535 for jtln in jump_to_line_number]):
            raise ValueError('One or more data generator subsequence/block repetitions are not between 1 and 65535.')

        if wait_on_trigger is None:
            wait_on_trigger = [0]*len(names)
        if not np.all([(wot == 0 or wot == 1) for wot in wait_on_trigger]):
            raise ValueError('One or more data generator subsequence/block wait_on_trigger values are neither 0 nor 1.')

        if event_jump is None:
            event_jump = [0]*len(names)
        if not np.all([(ej == 0 or ej == 1) for ej in event_jump]):
            raise ValueError('One or more data generator subsequence/block event_jump values are neither 0 nor 1.')

        if infinite_loop is None:
            infinite_loop = [1]*len(names)
        if not np.all([(il == 0 or il == 1) for il in infinite_loop]):
            raise ValueError('One or more data generator subsequence/block infinite_loop values are neither 0 nor 1.')

        # convert numbers to strings
        repetitions = [str(r) for r in repetitions]
        jump_to_line_number = [str(jtln) for jtln in jump_to_line_number]
        wait_on_trigger = [str(wot) for wot in wait_on_trigger]
        event_jump = [str(ej) for ej in event_jump]
        infinite_loop = [str(il) for il in infinite_loop]

        # construct a string of the type [subsequence_name],[block1name],[block1rep],[block2name],[block2rep]...
        subsequence_string = '\n'.join([names[i].upper() + ',' + repetitions[i] + ',' +  jump_to_line_number[i] + ',' +
                                       wait_on_trigger[i] + ',' + event_jump[i] + ',' + infinite_loop[i]
                                       for i in range(len(names))])

        definition_length = str(len(subsequence_string)) # the minus 1 is for the last \n which should not be counted
        definition_length_digits = str(len(definition_length))

        command = '\n' + self.sequence_define_cmd + ' #' + definition_length_digits + definition_length +\
                  subsequence_string + '\n'

        # return command
        return self.simple_write(command)

    def define_subsequence(self, name, block_names, repetitions):
        if not isinstance(name, str):
            raise TypeError('Data generator subsequence_name must be a string.')
        if isinstance(block_names, str):
            block_names = [block_names]
        if not np.all([isinstance(bn, str) for bn in block_names]):
            raise TypeError('Data generator subsequence block_names must be a string or a list of strings.')
        if isinstance(repetitions, int):
            repetitions = [repetitions]
        if not np.all([isinstance(bn, str) for bn in block_names]):
            raise TypeError('Data generator subsequence repetitions must be an integer or a list of integers.')
        if len(block_names) != len(repetitions):
            raise ValueError('Data generator subsequence block_names and repetition list lengths need to be equal.')

        # convert numbers to strings
        repetitions = [str(r) for r in repetitions]

        # construct a string of the type [subsequence_name],[block1name],[block1rep],[block2name],[block2rep]...
        subsequence_string = name + ',' + ','.join([block_names[i].upper() + ',' + repetitions[i]
                                                    for i in range(len(block_names))])

        definition_length = str(len(subsequence_string))
        definition_length_digits = str(len(definition_length))

        command = '\n' + self.subsequence_define_cmd + ' #' + definition_length_digits + definition_length +\
                  subsequence_string + '\n'

        return self.simple_write(command)

    def start(self):
        return self.simple_write(self.start_cmd)

    def stop(self):
        self.simple_write(self.stop_cmd)

        if self.need_to_set_initiazation_parameters:
            if self.stored_initialization_frequency_hz is not None:
                self.set_oscillator_internal_frequency(self.stored_initialization_frequency_hz, units='Hz')
            if self.stored_initialization_memory_size is not None:
                self.set_memory(self.stored_initialization_memory_size)
            self.need_to_set_initiazation_parameters = False  # once it does it once, there is no need to do it again
