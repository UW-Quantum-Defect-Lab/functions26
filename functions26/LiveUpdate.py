# 2021-3-28
# This code was made for use in the Fu lab
# by Vasilis Niaouris

import multiprocessing as mp
import numpy as np
import seaborn as sns
import threading
import time

from multiprocessing import shared_memory
from spinmob import egg
from .filing.QDLFiling import QDLFDataManager
from .units.UnitClass import UnitClass


class BasicGUIFunctions:

    def __init__(self, window_name='', font_size_reading=None, font_size_axes=None, plot_labels=None, no_of_datasets=1,
                 datatype=None):

        self.window_name = window_name
        self.font_size_reading = font_size_reading
        self.font_size_axes = font_size_axes
        self.plot_labels = plot_labels
        self.no_of_datasets = no_of_datasets
        self.datatype = datatype

        # initialize datasets as a dictionary with keys 'x1', 'y1', 'x2', 'y2' etc up to no_of_datasets enumeration
        self.datasets = {}
        for i in range(self.no_of_datasets):
            self.datasets['x' + str(i + 1)] = []
            self.datasets['y' + str(i + 1)] = []
        # setting datatype will determine how QDLFInterface will read the saved file.

        # initializing an empty parameter dictionary. Will be using dict to set the settings easier
        self.parameters = {}

        self.is_running = False

        self.window = egg.gui.Window(self.window_name, event_close=self.on_shutdown_click)

        # initializing setting tree-dictionary. May not be used in simple applications.
        settings_file_name = 'settings'
        if self.window_name != '':
            settings_file_name = self.window_name + ' ' + settings_file_name
        self.settings = egg.gui.TreeDictionary(autosettings_path=settings_file_name)

        self.start_button = egg.gui.Button("Start")
        self.start_button.signal_clicked.connect(self.on_start_click)
        self.current_start_click_count = 0

        self.stop_button = egg.gui.Button("Stop")
        self.stop_button.signal_clicked.connect(self.on_stop_click)

        self.save_button = egg.gui.Button("Save")
        self.save_button.signal_clicked.connect(self.on_save_click)

        self.clear_button = egg.gui.Button("Clear")
        self.clear_button.signal_clicked.connect(self.on_clear_click)

        self.shutdown_button = egg.gui.Button("Shutdown")
        self.shutdown_button.signal_clicked.connect(self.on_shutdown_click)

        self.apply_settings_changes_button = egg.gui.Button("Apply Changes")
        self.apply_settings_changes_button.signal_clicked.connect(self.on_apply_setting_changes_click)

        self.plot_widget = egg.pyqtgraph.PlotWidget()
        self.set_plot_widget_axis_style_and_labels()

        self.plot_curve = []
        for i in range(self.no_of_datasets):
            self.plot_curve.append(egg.pyqtgraph.PlotCurveItem())
            self.plot_widget.addItem(self.plot_curve[i])
        self.set_plot_curve_colors(color_cycle=sns.color_palette("tab10"))

        self.current_readings = []
        for i in range(self.no_of_datasets):
            self.current_readings.append(egg.gui.Label('NO DATA'))
        self.set_reading_font()

    def on_start_click(self):
        self.current_start_click_count += 1
        start_click_count_on_entry = self.current_start_click_count
        self.is_running = True
        self.start_button.disable()
        self.stop_button.enable()

        self.start_process(start_click_count_on_entry)

    def start_process(self, start_click_count_on_entry):
        print('Define your own start process.')

    def on_stop_click(self):
        self.is_running = False
        self.stop_button.disable()
        self.start_button.enable()

        self.stop_process()

    def stop_process(self):
        print('Define your own stop process.')

    def on_clear_click(self):
        was_running = self.is_running
        if was_running:
            self.stop_button.click()

        self.clear_process()

        if was_running:
            self.start_button.click()

    def clear_process(self):
        print('Define your own clear process.')

    def on_save_click(self):
        filename = egg.pyqtgraph.FileDialog.getSaveFileName(self.window._window, 'Save File')[0]
        qdlf_mng = self.get_data_manager_for_file_saving()
        qdlf_mng.save(filename)

    def get_data_manager_for_file_saving(self) -> QDLFDataManager:
        data = self.datasets
        # getting the length of the last key to avoid length inconsistencies while parallel process is running.
        last_key_length = -1
        for key in data:
            last_key_length = len(data[key])
        return QDLFDataManager(data={key: list(data[key][:last_key_length]) for key in data.keys()},
                               parameters=self.parameters, datatype=self.datatype)

    def on_shutdown_click(self):
        self.stop_button.click()
        self.shutdown_process()

    def shutdown_process(self):
        print('Define your own shutdown function if you need to terminate instruments')

    def on_apply_setting_changes_click(self):
        for key in self.settings.keys():
            if isinstance(self.parameters[key], UnitClass):
                self.parameters[key] = UnitClass(self.settings.get_value(key), self.parameters[key].original_unit)
            else:
                self.parameters[key] = self.settings.get_value(key)
        self.apply_setting_changes_process()

    def apply_setting_changes_process(self):
        print('Define your own apply_setting_changes process')

    def set_plot_widget_axis_style_and_labels(self):
        if self.font_size_axes is not None:
            self.plot_widget.getPlotItem()
            font = egg.pyqtgraph.QtGui.QFont()
            font.setPointSize(self.font_size_axes)
            label_style = {'color': '#FFF', 'font-size': str(self.font_size_axes) + 'pt'}
            for axis_name in ['left', 'bottom']:
                axis = self.plot_widget.getPlotItem().getAxis(axis_name)
                axis.setStyle(tickFont=font)
                if self.plot_labels is not None:
                    axis.setLabel(self.plot_labels[axis_name], **label_style)

    def set_plot_curve_colors(self, color_cycle):
        for i in range(self.no_of_datasets):
            # this is changing the plot line attributes (color, width etc).
            curve_pen = egg.pyqtgraph.mkPen(width=3, color=([crgb * 255 for crgb in color_cycle[i]]))
            self.plot_curve[i].setPen(curve_pen)

    def set_reading_font(self):
        if self.font_size_reading is not None:
            for i in range(self.no_of_datasets):
                self.current_readings[i].set_style('font-size: ' + str(self.font_size_reading) + 'px')


class LiveUpdateFunctions(BasicGUIFunctions):
    form_dataset_process_methods_available = ['averaging', 'last_point', 'spcm', 'user']
    plot_normalization_methods_available = ['none', 'per_second']

    def __init__(self, plot_labels=None, no_of_datasets=1, maximum_x_axis_points=0, run_on_call=False,
                 gui_refresh_rate=UnitClass(20, 'Hz'), device_acquisition_rate=UnitClass(50, 'Hz'),
                 form_dataset_process_method='averaging', plot_normalization_method='none',
                 current_reading_form='{:,.3f}', current_reading_units='', datatype=None, window_name='Live Update'):

        if plot_labels is None:
            plot_labels = {'bottom': 'Time (seconds)', 'left': 'Reading'}

        super().__init__(window_name=window_name, font_size_reading=100, font_size_axes=25, plot_labels=plot_labels,
                         no_of_datasets=no_of_datasets, datatype=datatype)

        self.maximum_x_axis_points = maximum_x_axis_points

        if isinstance(gui_refresh_rate, UnitClass):
            self.starting_gui_refresh_rate = gui_refresh_rate
        else:
            self.starting_gui_refresh_rate = UnitClass(gui_refresh_rate, 'Hz')

        if isinstance(device_acquisition_rate, UnitClass):
            self.parameters["Acquisition/Rate"] = device_acquisition_rate
        else:
            self.parameters["Acquisition/Rate"] = UnitClass(device_acquisition_rate, 'Hz')

        self.form_dataset_process_method = form_dataset_process_method
        self.plot_normalization_method = plot_normalization_method
        self.current_reading_form = current_reading_form
        self.current_reading_units = current_reading_units

        # gui can not refresh faster than the data is produced
        if self.starting_gui_refresh_rate > self.parameters["Acquisition/Rate"]:
            self.starting_gui_refresh_rate = self.parameters["Acquisition/Rate"]

        # the amount of data we will be sampling over to average
        self.averaging_sample_number = self.get_averaging_sample_number()
        self.parameters["GUI/Rate"] = self.get_adjusted_gui_refresh_rate()
        print('Set gui refresh rate to ' + str(self.parameters["GUI/Rate"].Hz) + ' Hz so the device_acquisition_rate is'
                                                                                 ' a multiple of the former rate.')

        if self.form_dataset_process_method == self.form_dataset_process_methods_available[0]:
            self.form_datasets_process = self.form_datasets_process_averaging
        elif self.form_dataset_process_method == self.form_dataset_process_methods_available[1]:
            self.form_datasets_process = self.form_datasets_process_last_point
        elif self.form_dataset_process_method == self.form_dataset_process_methods_available[2]:
            self.form_datasets_process = self.form_datasets_process_spcm
        elif self.form_dataset_process_method == self.form_dataset_process_methods_available[3]:
            self.form_datasets_process = self.form_datasets_process_user
        else:
            raise ValueError('Form dataset process method not recognised. Choose among the following:\n'
                             + str(self.form_dataset_process_methods_available))

        if self.plot_normalization_method == self.plot_normalization_methods_available[0]:
            self.plot_normalization_multiplier = 1
        elif self.plot_normalization_method == self.plot_normalization_methods_available[1]:
            self.plot_normalization_multiplier = self.parameters["GUI/Rate"].Hz

        # initializing shared memory variables
        self.shared_memory_instance = None
        self.shared_memory_name = None
        self.shared_memory_buffer = None
        # create shared memory
        self.create_shared_memory()
        self.update_shared_memory_buffer()

        # defining multiprocessing manager
        self.multiprocessing_manager = mp.Manager()
        # defining Queque that will transfer data from the acquisition process to the forming datasets process
        self.data_queue = self.multiprocessing_manager.Queue()
        # changing the datasets to a form that is transferable between the form_datasets_process and the gui
        self.datasets = self.multiprocessing_manager.dict(self.datasets)
        # now every list in the datasets need to be restated as a list in the manager, otherwise data will
        # not be appended. This method only works with 3.6 python or later versions.
        for key in self.datasets.keys():
            self.datasets[key] = self.multiprocessing_manager.list(self.datasets[key])

        self.start_time = time.time()

        # getting any user defined parameters
        self.define_user_parameters()
        self.before_run_on_call()

        if run_on_call:
            self.start_button.click()
        else:
            self.stop_button.disable()
            self.start_button.enable()

    def define_user_parameters(self):
        # This fucntion can be overwritten to define parameters other than GUI and Acquisition Rates
        pass

    def before_run_on_call(self):
        # This function can be overwritten to run other code before starting on run on call
        pass

    def start_process(self, start_click_count_on_entry):
        if self.shared_memory_instance is None:
            self.create_shared_memory()
        self.update_shared_memory_buffer()

        self.dap = mp.Process(target=self.data_acquisition_process,
                              args=(self.data_queue, self.parameters, start_click_count_on_entry,
                                    self.shared_memory_name, self.start_time))
        self.fdp = mp.Process(target=self.form_datasets_process,
                              args=(self.datasets, self.data_queue, self.averaging_sample_number,
                                    start_click_count_on_entry, self.shared_memory_name, self.parameters))

        putp = threading.Thread(target=self.plot_update_thread_process, args=(start_click_count_on_entry,))
        crutp = threading.Thread(target=self.current_reading_update_thread_process, args=(start_click_count_on_entry,))

        putp.daemon = True
        crutp.daemon = True

        if self.user_pre_start_process_check():
            putp.start()
            crutp.start()
            self.dap.start()
            self.fdp.start()
        else:
            self.stop_button.click()

    def user_pre_start_process_check(self) -> bool:
        # This function can be overwritten by the user to provide in case user wants conditions to start
        # processes and threads
        return True

    def stop_process(self):
        self.update_shared_memory_buffer()
        # if self.dap.is_alive():
        #     self.dap.kill()
        #     self.fdp.kill()
        #     self.destroy_shared_memory_link()
        self.user_post_stop_process()

    def user_post_stop_process(self):
        # This function can be overwritten by the user to provide some sort of input after the stopping process
        pass

    def clear_process(self):
        self.start_time = time.time()
        for i in range(self.no_of_datasets):
            self.datasets['x' + str(i + 1)] = self.multiprocessing_manager.list([])
            self.datasets['y' + str(i + 1)] = self.multiprocessing_manager.list([])
            self.plot_curve[i].setData([0], [0])

    def shutdown_process(self):
        print('Does not do more than stop at this point. Define your own if you want more done')

    def update_shared_memory_buffer(self):
        self.shared_memory_buffer[0] = self.is_running
        self.shared_memory_buffer[1] = self.current_start_click_count

    def create_shared_memory(self):
        self.shared_memory_instance = shared_memory.SharedMemory(create=True, size=2)
        self.shared_memory_name = self.shared_memory_instance.name
        self.shared_memory_buffer = self.shared_memory_instance.buf

    def destroy_shared_memory_link(self):
        self.shared_memory_instance.close()
        self.shared_memory_instance.unlink()
        self.shared_memory_instance = None
        self.shared_memory_name = None
        self.shared_memory_buffer = None

    def get_averaging_sample_number(self):
        return int(np.floor(self.parameters["Acquisition/Rate"] / self.starting_gui_refresh_rate))

    def get_adjusted_gui_refresh_rate(self):
        return UnitClass(self.parameters["Acquisition/Rate"].Hz / self.averaging_sample_number, 'Hz')

    @staticmethod
    def data_acquisition_process(queue, parameters, start_click_count_on_entry, shared_memory_name, start_time):
        print('Define your own data_acquisition_process')

    @staticmethod
    def form_datasets_process_averaging(datasets, data_queue, averaging_sample_number, start_click_count_on_entry,
                                        shared_memory_name, parameters):

        existing_shared_memory = shared_memory.SharedMemory(name=shared_memory_name)
        existing_shared_memory_buffer = existing_shared_memory.buf
        while existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
            temp = {key: [] for key in datasets.keys()}
            for i in range(averaging_sample_number):
                if existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
                    queue_element = data_queue.get()
                    for key in datasets.keys():
                        temp[key].append(queue_element[key])
            if existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
                for key in datasets.keys():
                    datasets[key].append(np.mean(temp[key]))

        existing_shared_memory.close()

    @staticmethod
    def form_datasets_process_last_point(datasets, data_queue, averaging_sample_number, start_click_count_on_entry,
                                         shared_memory_name, parameters):
        existing_shared_memory = shared_memory.SharedMemory(name=shared_memory_name)
        existing_shared_memory_buffer = existing_shared_memory.buf
        while existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
            for i in range(averaging_sample_number):
                if existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
                    queue_element = data_queue.get()
                    if i == averaging_sample_number - 1:
                        for key in datasets.keys():
                            datasets[key].append(queue_element[key])
        existing_shared_memory.close()

    @staticmethod
    def form_datasets_process_spcm(datasets, data_queue, averaging_sample_number, start_click_count_on_entry,
                                   shared_memory_name, parameters):
        gui_rate = parameters['GUI/Rate'].Hz

        last_points = np.zeros(int(len(datasets) / 2))
        new_points = np.zeros(int(len(datasets) / 2))
        gui_time_step = 1 / gui_rate
        existing_shared_memory = shared_memory.SharedMemory(name=shared_memory_name)
        existing_shared_memory_buffer = existing_shared_memory.buf
        while existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
            for i in range(averaging_sample_number):
                if existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
                    queue_element = data_queue.get()
                    if i == averaging_sample_number - 1:
                        for j in range(int(len(datasets) / 2)):
                            keyx = 'x' + str(j + 1)
                            keyy = 'y' + str(j + 1)
                            datasets[keyx].append(queue_element[keyx])
                            if len(datasets[keyy]):  # that means not 0
                                dt = datasets[keyx][-1] - datasets[keyx][-2]
                                no_of_steps = np.round(dt / gui_time_step)
                                new_points[j] = queue_element[keyy]
                                datasets[keyy].append((new_points[j] - last_points[j]) / no_of_steps)
                                last_points[j] = new_points[j]
                            else:  # the one time this will be zero
                                new_points[j] = queue_element[keyy]
                                datasets[keyy].append(new_points[j])
                                last_points[j] = new_points[j]

        existing_shared_memory.close()

    @staticmethod
    def form_datasets_process_user(datasets, data_queue, averaging_sample_number, start_click_count_on_entry,
                                   shared_memory_name, parameters):
        print('Define your own form_datasets_process_user. Defaulting to using the form_datasets_process_averaging')
        # here we copy the process in case the user did not define their own.
        existing_shared_memory = shared_memory.SharedMemory(name=shared_memory_name)
        existing_shared_memory_buffer = existing_shared_memory.buf
        while existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
            temp = {key: [] for key in datasets.keys()}
            for i in range(averaging_sample_number):
                if existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
                    queue_element = data_queue.get()
                    for key in datasets.keys():
                        temp[key].append(queue_element[key])
            if existing_shared_memory_buffer[0] and start_click_count_on_entry == existing_shared_memory_buffer[1]:
                for key in datasets.keys():
                    datasets[key].append(np.mean(temp[key]))

        existing_shared_memory.close()

    def plot_update_thread_process(self, start_click_count_on_entry):
        while self.is_running and start_click_count_on_entry == self.current_start_click_count:
            local_datasets = dict(self.datasets)
            for i in range(self.no_of_datasets):
                x = local_datasets['x' + str(i + 1)]
                x = np.array(x[-self.maximum_x_axis_points:len(x)])
                y = local_datasets['y' + str(i + 1)]
                y = np.array(y[-self.maximum_x_axis_points:len(y)]) * self.plot_normalization_multiplier
                if len(x) == len(y):
                    # sometimes, you might get an exception that the lengths of the locally defined variables are not
                    # the same, even after the if statement. I think it might have to do with the datasets being a
                    # manager dictionary that changes from the data processing, but I am not sure exactly why. This
                    # definition of x and y seems promising.
                    self.plot_curve[i].setData(x, y)
            time.sleep(1 / self.parameters["GUI/Rate"])

    def current_reading_update_thread_process(self, start_click_count_on_entry):
        while self.is_running and start_click_count_on_entry == self.current_start_click_count:
            local_datasets = self.datasets
            for i in range(self.no_of_datasets):
                y = local_datasets['y' + str(i + 1)]
                if len(y) > 1:
                    self.current_readings[i].set_text(
                        self.current_reading_form.format(y[-1] * self.plot_normalization_multiplier) +
                        self.current_reading_units)
            time.sleep(1 / self.parameters["GUI/Rate"])


class LiveUpdateGUI(LiveUpdateFunctions):

    def __init__(self, plot_labels=None, no_of_datasets=1, maximum_x_axis_points=0, run_on_call=False,
                 gui_refresh_rate=UnitClass(20, 'Hz'), device_acquisition_rate=UnitClass(50, 'Hz'),
                 form_dataset_process_method='averaging', plot_normalization_method='none',
                 current_reading_form='{:.3f}', current_reading_units='', datatype=None, window_name='Live Update'):
        super().__init__(plot_labels=plot_labels, no_of_datasets=no_of_datasets,
                         maximum_x_axis_points=maximum_x_axis_points, run_on_call=run_on_call,
                         gui_refresh_rate=gui_refresh_rate, device_acquisition_rate=device_acquisition_rate,
                         form_dataset_process_method=form_dataset_process_method,
                         plot_normalization_method=plot_normalization_method, current_reading_form=current_reading_form,
                         current_reading_units=current_reading_units, datatype=datatype, window_name=window_name)

        self.window.place_object(self.start_button, 0, 0)
        self.window.place_object(self.stop_button, 1, 0)
        self.window.place_object(self.save_button, 2, 0)
        self.window.place_object(self.clear_button, 3, 0, alignment=2)
        self.window.place_object(self.shutdown_button, 4, 0, alignment=2)
        self.window.set_column_stretch(2, 20)  # creates big space between save and shutdown buttons
        self.window.place_object(self.plot_widget, 0, 1, column_span=5, alignment=0)

        for i in range(self.no_of_datasets):
            self.window.place_object(self.current_readings[i], (i % 2) * 2, int(np.floor(2 + i / 2)), column_span=3,
                                     alignment=i % 2 + 1)


class AdvancedLiveUpdateGUI(LiveUpdateFunctions):

    def __init__(self, plot_labels=None, no_of_datasets=1, maximum_x_axis_points=0, run_on_call=False,
                 gui_refresh_rate=UnitClass(20, 'Hz'), device_acquisition_rate=UnitClass(50, 'Hz'),
                 form_dataset_process_method='averaging', plot_normalization_method='none',
                 current_reading_form='{:.3f}', current_reading_units='', datatype=None, window_name='Live Update'):

        super().__init__(plot_labels=plot_labels, no_of_datasets=no_of_datasets,
                         maximum_x_axis_points=maximum_x_axis_points, run_on_call=run_on_call,
                         gui_refresh_rate=gui_refresh_rate, device_acquisition_rate=device_acquisition_rate,
                         form_dataset_process_method=form_dataset_process_method,
                         plot_normalization_method=plot_normalization_method, current_reading_form=current_reading_form,
                         current_reading_units=current_reading_units, datatype=datatype, window_name=window_name)

        # self.window.set_width(900)

        # self.tab_space_1 = egg.gui.TabArea(autosettings_path='tab_space_1')
        self.tab_space_1 = egg.gui.TabArea()
        self.settings_tab = self.tab_space_1.add_tab('Settings')

        # self.tab_space_2 = egg.gui.TabArea(autosettings_path='tab_space_2')
        self.tab_space_2 = egg.gui.TabArea()
        self.data_collection_tab = self.tab_space_2.add_tab('Data collection')

        self.window.place_object(self.tab_space_2, column=2, column_span=5)
        self.data_collection_tab.place_object(self.start_button, 0, 0)
        self.data_collection_tab.place_object(self.stop_button, 1, 0)
        self.data_collection_tab.place_object(self.save_button, 2, 0)
        self.data_collection_tab.place_object(self.clear_button, 3, 0, alignment=2)
        self.data_collection_tab.place_object(self.shutdown_button, 4, 0, alignment=2)
        self.data_collection_tab.set_column_stretch(2, 20)  # creates big space between save and shutdown buttons
        self.data_collection_tab.place_object(self.plot_widget, 0, 1, column_span=5, alignment=0)

        for i in range(self.no_of_datasets):
            self.data_collection_tab.place_object(self.current_readings[i], (i % 2) * 2, int(np.floor(2 + i / 2)),
                                                  column_span=3, alignment=i % 2 + 1)

        self.window.place_object(self.tab_space_1, row=0, column=0, column_span=2)
        self.settings_tab.place_object(self.apply_settings_changes_button, row=0, column=0, alignment=0)
        self.settings_tab.place_object(self.settings, row=1, column=0, column_span=10, row_span=5, alignment=0)
        self.settings_tab.set_column_stretch(2, 10)  # creates big space between apply changes button and the
        # empty space on its right

        self.settings.add_parameter("Acquisition/Rate", type='float', bounds=(1.e-2, 1.e5), siPrefix=True, suffix='Hz')
        self.settings.set_value("Acquisition/Rate", self.parameters["Acquisition/Rate"])

        self.settings.add_parameter("GUI/Rate", type='float', bounds=(1.e-2, 1.e3), siPrefix=True, suffix='Hz')
        self.settings.set_value("GUI/Rate", self.parameters["GUI/Rate"])
        # self.settings.connect_signal_changed("GUI/Rate", self.gui_refresh_rate_setting_change_action)

    def apply_setting_changes_process(self):
        # self.parameters["Acquisition/Rate"] = UnitClass(self.settings.get_value("Acquisition/Rate"), 'Hz')
        self.starting_gui_refresh_rate = UnitClass(self.settings.get_value("GUI/Rate"), 'Hz')
        self.adjust_gui_refresh_rate_in_settings()
        self.adding_users_apply_setting_changes_process()
        if self.is_running:
            self.stop_button.click()
            self.start_button.click()

    def adjust_gui_refresh_rate_in_settings(self):
        if self.starting_gui_refresh_rate.Hz > self.parameters["Acquisition/Rate"].Hz:
            self.starting_gui_refresh_rate = self.parameters["Acquisition/Rate"]
        self.averaging_sample_number = self.get_averaging_sample_number()
        self.parameters["GUI/Rate"] = self.get_adjusted_gui_refresh_rate()
        if self.parameters["GUI/Rate"].Hz != self.settings.get_value("GUI/Rate"):
            self.settings.set_value("GUI/Rate", self.parameters["GUI/Rate"].Hz)
            print('Set gui refresh rate to ' + str(self.parameters["GUI/Rate"].Hz) + 'Hz so the '
                                                                                     'device_acquisition_rate is a '
                                                                                     'multiple of the former rate.')

    def define_user_settings(self):
        # This function can be overwritten to define user settings/parameters before run_on_call of parent class
        pass

    def before_run_on_call(self):
        self.define_user_settings()

    def adding_users_apply_setting_changes_process(self):
        # This is function to be overwritten by the user to make other changes to the parameters when settings
        # are applied
        pass
