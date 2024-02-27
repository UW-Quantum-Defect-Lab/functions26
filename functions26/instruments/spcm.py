import numpy as np
import threading
import time
import queue

from functions26.filing.QDLFiling import QDLFDataManager
from functions26.InstrumentHandler import GPIBInstrument, NIdaqInstrument
from functions26.units.UnitClass import UnitClass
from functools import partial
from nidaqmx import stream_readers as ndsr
from nidaqmx.constants import Edge, AcquisitionType, ReadRelativeTo


class SPCM:

    spcm_name = 'dev1/ctr1'
    internal_rate = UnitClass(20000000, 'Hz')  # Hz

    def __init__(self, instrument_port=None):
        if instrument_port is not None:
            self.spcm_name = instrument_port
        self.spcm = NIdaqInstrument(self.spcm_name)
        self.reader = None
        self.reading_spcm_task_callback = None
        self.pipeline = queue.Queue()
        self.thread = None

    def start_acquisition(self, start_time: float, start_event: threading.Event, stop_event: threading.Event,
                          sleep_time: float = 0.05):
        self.spcm.initialize_instrument()

        self.spcm.task.timing.cfg_samp_clk_timing(self.internal_rate, source="/dev1/20MHzTimebase",
                                                  active_edge=Edge.RISING, sample_mode=AcquisitionType.CONTINUOUS)

        self.spcm.task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
        self.spcm.task.in_stream.offset = -1

        self.reader = ndsr.CounterReader(self.spcm.task.in_stream)

        def reading_spcm_task_callback(data, t0, task_idx, every_n_samples_event_type, num_samples, callback_data):
            # Read values from spcm and append to spcm_data
            buffer = np.zeros(1)
            self.reader.read_many_sample_double(buffer, number_of_samples_per_channel=1)
            data.append([time.time() - t0, buffer[0]])
            return 0

        self.reading_spcm_task_callback = reading_spcm_task_callback

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
            self.spcm.terminate_instrument()
            del self.reader
            self.reader = None
            del self.reading_spcm_task_callback
            self.reading_spcm_task_callback = None
            del self.thread
            self.thread = None

        return data_manager

    @staticmethod
    def save_acquisition(filename, data_manager: QDLFDataManager):
        data_manager.save(filename)

    def acquisition(self, start_time: float, time_step: float, pipeline: queue.Queue, start_event: threading.Event,
                    stop_event: threading.Event):

        sample_no = int(self.internal_rate.Hz * time_step)
        spcm_data = []
        self.spcm.task.register_every_n_samples_acquired_into_buffer_event(sample_no,
                                                                           partial(self.reading_spcm_task_callback,
                                                                                   spcm_data, start_time))
        while not start_event.is_set():
            time.sleep(time_step)

        self.spcm.task.start()
        while not stop_event.is_set():
            time.sleep(time_step)
        self.spcm.task.stop()

        spcm_data = np.transpose(spcm_data)
        data = {'x1': spcm_data[0][1:], 'y1': np.diff(spcm_data[1])}

        data_manager = QDLFDataManager(data, parameters={'start_time': start_time, 'time_step': time_step},
                                       datatype='spcm')
        pipeline.put(data_manager)

    def is_available(self):
        try:
            self.spcm.initialize_instrument()
            self.spcm.terminate_instrument()
            return True
        except Exception:
            return False

    def shutdown(self):
        self.stop_acquisition()
