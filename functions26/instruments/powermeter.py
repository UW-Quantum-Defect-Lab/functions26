
import numpy as np
import threading
import time
import queue

from functions26.filing.QDLFiling import QDLFDataManager
from functions26.InstrumentHandler import GPIBInstrument


class PowerMeter:
    powermeter_name = 'GPIB0::5::INSTR'
    read_termination = '\n'
    read_commandA = 'R_A?'
    read_commandB = 'R_B?'

    def __init__(self, channel: str = 'B'):
        self.channel = channel
        self.channel_commands = {'A': [self.read_commandA], 'B': [self.read_commandB], 'AB': [self.read_commandA,
                                                                                          self.read_commandB]}
        self.read_commands = self.channel_commands[channel]
        self.powermeter = GPIBInstrument(self.powermeter_name, self.read_termination, self.read_commands,
                                         initialize_at_definition=False)
        self.pipeline = queue.Queue()
        self.thread = None

    @staticmethod
    def convert_reading_string_to_float(reading_string):
        return float(reading_string) * 10 ** 6

    def _empty_buffer(self):

        buffer_empty = False
        while not buffer_empty:
            try:
                print(self.powermeter.simple_read())
                time.sleep(0.01)
            except Exception:
                buffer_empty = True

    def start_acquisition(self, start_time: float, start_event: threading.Event, stop_event: threading.Event,
                          sleep_time: float = 0.05):
        self.powermeter.initialize_instrument()
        self._empty_buffer()
        self.thread = threading.Thread(target=self.acquisition, args=(start_time, sleep_time, self.pipeline,
                                                                      start_event, stop_event), daemon=True)
        self.thread.start()

        return True

    def stop_and_save_acquisition(self, filename):
        data_manager = self.stop_acquisition()
        self.save_acquisition(filename, data_manager)

    def stop_acquisition(self):
        data_manager = QDLFDataManager()  # Empty manager
        if self.thread is not None:
            self.thread.join()
            data_manager: QDLFDataManager = self.pipeline.get()
            self.powermeter.terminate_instrument()
            del self.thread
            self.thread = None

        return data_manager

    @staticmethod
    def save_acquisition(filename, data_manager: QDLFDataManager):
        data_manager.save(filename)

    def acquisition(self, start_time: float, sleep_time: float, pipeline: queue.Queue, start_event: threading.Event,
                    stop_event: threading.Event):
        time_array = np.array([], dtype=float)
        power_list = []
        while not start_event.is_set():
            time.sleep(sleep_time)

        while not stop_event.is_set():
            new_time = time.time()
            try:
                reading_strings = self.powermeter.get_instrument_reading_string_all()
                readings = []
                for reading_string in reading_strings:
                    reading = self.convert_reading_string_to_float(reading_string)
                    readings.append(reading)
                power_list.append(readings)
                time_array = np.append(time_array, new_time)
            except Exception as e:
                pass
            time.sleep(sleep_time)

        data = {}
        if len(self.read_commands) == 1:
            data['x1'] = time_array-start_time
            data['y1'] = np.reshape(power_list, len(power_list))
        elif len(self.read_commands) > 1:
            power_array = np.transpose(power_list)
            for i in range(len(self.read_commands)):
                data[f'x{i+1}'] = time_array-start_time
                data[f'y{i+1}'] = power_array[i]
        else:
            data['x1'] = []
            data['y1'] = []

        data_manager = QDLFDataManager(data, parameters={'start_time': start_time, 'sleep_time': sleep_time},
                                       datatype='power')
        pipeline.put(data_manager)

    def is_available(self):
        try:
            self.powermeter.initialize_instrument()
            self.powermeter.terminate_instrument()
            return True
        except Exception:
            return False

