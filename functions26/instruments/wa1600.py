
import numpy as np
import threading
import time
import queue

from functions26.filing.QDLFiling import QDLFDataManager
from functions26.InstrumentHandler import GPIBInstrument


class WA1600:
    wa1600_name = 'GPIB0::18::INSTR'
    read_termination = '\n'
    read_command = ':READ:WAV?'

    def __init__(self, instrument_port=None):
        if instrument_port is not None:
            self.wa1600_name = instrument_port
        self.wa1600 = GPIBInstrument(self.wa1600_name, self.read_termination, self.read_command,
                                     initialize_at_definition=False)
        self.pipeline = queue.Queue()
        self.thread = None

    @staticmethod
    def convert_reading_string_to_float(reading_string):
        return float(reading_string)

    def _empty_buffer(self):

        buffer_empty = False
        while not buffer_empty:
            try:
                print(self.wa1600.simple_read())
                time.sleep(0.01)
            except Exception:
                buffer_empty = True

    def start_acquisition(self, start_time: float, start_event: threading.Event, stop_event: threading.Event,
                          sleep_time: float = 0.05):
        self.wa1600.initialize_instrument()
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
            self.wa1600.terminate_instrument()
            del self.thread
            self.thread = None

        return data_manager

    @staticmethod
    def save_acquisition(filename, data_manager: QDLFDataManager):
        data_manager.save(filename)

    def acquisition(self, start_time: float, sleep_time: float, pipeline: queue.Queue, start_event: threading.Event,
                    stop_event: threading.Event):
        time_array = np.array([], dtype=float)
        wavelength_list = []
        while not start_event.is_set():
            time.sleep(sleep_time)

        while not stop_event.is_set():
            new_time = time.time()
            try:
                reading_string = self.wa1600.get_instrument_reading()
                reading = self.convert_reading_string_to_float(reading_string)
                wavelength_list.append(reading)
                time_array = np.append(time_array, new_time)
            except Exception:
                pass
            time.sleep(sleep_time)

        data = {}
        if len(self.read_commands) == 1:
            data['x1'] = time_array-start_time
            data['y1'] = np.array(wavelength_list)

        data_manager = QDLFDataManager(data, parameters={'start_time': start_time, 'sleep_time': sleep_time},
                                       datatype='wavelength')
        pipeline.put(data_manager)

    def is_available(self):
        try:
            self.wa1600.initialize_instrument()
            self.wa1600.terminate_instrument()
            return True
        except Exception:
            return False

    def shutdown(self):
        self.stop_acquisition()
