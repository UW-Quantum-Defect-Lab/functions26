# 2019-05-12 / last updated on 2020-09-15
# This code was made for use in the Fu lab
# by Vasilis Niaouris

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase


plt.rcParams["toolbar"] = "toolmanager"


class ToolbarModifier:

    def __init__(self, figure):
        self.figure = figure
        self.modify_toolbar()

        self.running = 1
        self.communicate = True
        self.refresh_now = False
        self.stop_now = False
        self.shut_down_now = False
        self.restart_now = False
        self.start_of_standby = None

    class PauseTool(ToolBase):

        def __init__(self, *args, toolbar_modifier, **kwargs):
            self.toolbar_modifier = toolbar_modifier  # by creating the tool, we give its own self.lu the whole lu class.
            super().__init__(*args, **kwargs)  # I don't know why this is needed

        def trigger(self, *args, **kwargs):
            self.toolbar_modifier.running = 0  # the running value is part of self(aka PauseTool).toolbar_modifier variable

    class ResumeTool(ToolBase):

        def __init__(self, *args, toolbar_modifier, **kwargs):
            self.toolbar_modifier = toolbar_modifier
            super().__init__(*args, **kwargs)

        def trigger(self, *args, **kwargs):
            self.toolbar_modifier.communicate = True
            self.toolbar_modifier.running = 1

    class RefreshTool(ToolBase):

        def __init__(self, *args, toolbar_modifier, **kwargs):
            self.toolbar_modifier = toolbar_modifier
            super().__init__(*args, **kwargs)

        def trigger(self, *args, **kwargs):
            self.toolbar_modifier.refresh_now = True
            self.toolbar_modifier.running = 2  # to make sure program doesnt crush if device is closed
            self.toolbar_modifier.communicate = True

    class StopTool(ToolBase):

        def __init__(self, *args, toolbar_modifier, **kwargs):
            self.toolbar_modifier = toolbar_modifier
            super().__init__(*args, **kwargs)

        def trigger(self, *args, **kwargs):
            self.toolbar_modifier.stop_now = True
            self.toolbar_modifier.running = 0
            self.toolbar_modifier.communicate = False

    class StartTool(ToolBase):

        def __init__(self, *args, toolbar_modifier, **kwargs):
            self.toolbar_modifier = toolbar_modifier
            super().__init__(*args, **kwargs)

        def trigger(self, *args, **kwargs):
            self.toolbar_modifier.restart_now = True
            self.toolbar_modifier.running = 2  # to re open the communication session with the device
            self.toolbar_modifier.communicate = True

    class ShutDownTool(ToolBase):

        def __init__(self, *args, toolbar_modifier, **kwargs):
            self.toolbar_modifier = toolbar_modifier
            super().__init__(*args, **kwargs)

        def trigger(self, *args, **kwargs):
            self.toolbar_modifier.shut_down_now = True
            self.toolbar_modifier.running = 0
            self.toolbar_modifier.communicate = False

    def modify_toolbar(self):
        figure = self.figure  # for some reason I can't directly fo self.figure.canvas...
        tm = figure.canvas.manager.toolmanager

        # Removing unwanted tools
        tm.remove_tool('back')
        tm.remove_tool('forward')
        tm.remove_tool('home')
        tm.remove_tool('pan')
        tm.remove_tool('zoom')

        # Adding tools
        tm.add_tool('Pause', self.PauseTool, toolbar_modifier=self)  # within the LU class, self is LiveUpdate ;)
        figure.canvas.manager.toolbar.add_tool(tm.get_tool('Pause'), "toolgroup")
        tm.add_tool('Resume', self.ResumeTool, toolbar_modifier=self)
        figure.canvas.manager.toolbar.add_tool(tm.get_tool('Resume'), "toolgroup")
        tm.add_tool('Refresh', self.RefreshTool, toolbar_modifier=self)
        figure.canvas.manager.toolbar.add_tool(tm.get_tool('Refresh'), "toolgroup")
        tm.add_tool('Stop', self.StopTool, toolbar_modifier=self)
        figure.canvas.manager.toolbar.add_tool(tm.get_tool('Stop'), "toolgroup")
        tm.add_tool('Start', self.StartTool, toolbar_modifier=self)
        figure.canvas.manager.toolbar.add_tool(tm.get_tool('Start'), "toolgroup")
        tm.add_tool('Shutdown', self.ShutDownTool, toolbar_modifier=self)
        figure.canvas.manager.toolbar.add_tool(tm.get_tool('Shutdown'), "toolgroup")


class AnimationManager:

    def __init__(self, toolbar_modifier, instrument, subplot_axis, data_line, text_box=None, standby_time=30,
                 start_time=time.time(), device_string_to_float=None, change_text_box_value=None):

        self.toolbar_modifier = toolbar_modifier
        self.instrument = instrument
        self.subplot_axis = subplot_axis
        self.data_line = data_line
        self.text_box = text_box
        self.standby_time = standby_time
        self.start_time = start_time
        if device_string_to_float is not None:
            self.device_string_to_float = device_string_to_float
        if change_text_box_value is not None:
            self.change_text_box_value = change_text_box_value

        self.waiting_for_standby = None
        self.reading_string = None

    def check_status(self):

        if self.toolbar_modifier.refresh_now:
            self.toolbar_modifier.refresh_now = False
            self.subplot_axis.cla()
            self.data_line, = self.subplot_axis.plot([], [])

        if self.toolbar_modifier.restart_now:
            self.toolbar_modifier.restart_now = False
            self.subplot_axis.cla()
            self.data_line, = self.subplot_axis.plot([], [])
            self.start_time = time.time()

        if self.toolbar_modifier.stop_now:
            self.toolbar_modifier.stop_now = False
            self.data_line, = self.subplot_axis.plot([], [])

        return self.data_line

    # def check_status_multi(self, data_lines, subplot_axes):
    #
    #     if self.refresh_now:
    #         self.refresh_now = False
    #         for i in range(len(data_lines)):
    #             subplot_axes[i].cla()
    #             data_lines[i], = subplot_axes[i].plot([], [])
    #
    #     if self.restart_now:
    #         self.restart_now = False
    #         for i in range(len(data_lines)):
    #             subplot_axes[i].cla()
    #             data_lines[i], = subplot_axes[i].plot([], [])
    #
    #         self.start_time = time.time()
    #
    #     if self.stop_now:
    #         self.stop_now = False
    #         for i in range(len(data_lines)):
    #             subplot_axes[i].cla()
    #             data_lines[i], = subplot_axes[i].plot([], [])
    #
    #     return data_lines

    def exec_standby_action(self):
        if self.text_box is not None:
            self.text_box.set_text('Standby. Click on graph to continue.')
            plt.ginput(1)  # so it won't go through loops for long times
            self.text_box.set_text('Press Res/Ref/Start.')
        else:
            print('Standby. Click on graph to continue.')
            plt.ginput(1)  # so it won't go through loops for long times
            print('Press Res/Ref/Start.')

    def exec_no_running_action(self):

        if not self.toolbar_modifier.communicate and self.waiting_for_standby is None:
            self.instrument.terminate_instrument()

        if self.waiting_for_standby is None:
            self.waiting_for_standby = time.time()
        elif time.time() - self.waiting_for_standby > self.standby_time:
            print('Automatic click standby')
            self.instrument.terminate_instrument()
            self.exec_standby_action()
            print('Click standby canceled')
            self.waiting_for_standby = None

        time.sleep(self.instrument.time_delay)

    def update_line(self, new_x, new_y):

        self.data_line.set_xdata(np.append(self.data_line.get_xdata(), new_x))
        self.data_line.set_ydata(np.append(self.data_line.get_ydata(), new_y))
        plt.draw()
        figure = self.toolbar_modifier.figure
        figure.canvas.flush_events()

    def get_data_point(self):

        try:
            self.reading_string = self.instrument.get_instrument_reading_string()
        except Exception as e:
            self.reading_string = None
        reading = self.device_string_to_float(self.reading_string)
        if reading is not None:
            time_now = time.time() - self.start_time
            self.update_line(time_now, reading)
            self.subplot_axis.relim()
            self.subplot_axis.autoscale()

            return reading, time_now
        else:
            return None, None

    def device_string_to_float(self, reading_string):
        reading_string = self.reading_string
        try:
            return float(reading_string)
        except ValueError:
            return None

    def change_text_box_value(self, value):
        if self.text_box is not None:
            print(value)

    def animation_loop(self, i):

        self.data_line = self.check_status()

        if self.toolbar_modifier.running == 1:
            data_point = self.get_data_point()
            self.change_text_box_value(data_point[0])

        elif self.toolbar_modifier.running == 2:
            self.instrument.reopening_session()
            self.toolbar_modifier.running = 1

            data_point = self.get_data_point()
            self.change_text_box_value(data_point[0])

        elif self.toolbar_modifier.running == 0:
            self.exec_no_running_action()
            if self.toolbar_modifier.shut_down_now:
                self.instrument.terminate_instrument()
                plt.close()


class CreateSubplotAndTextBox:

    def __init__(self, figure, reading_units_string, reading_what_string, default_subplot_creation=True):
        self.figure = figure
        self.reading_units_string = reading_units_string
        self.reading_what_string = reading_what_string

        # you may change those outside, in your code
        self.x_text = 0.05
        self.y_text = 0.5
        self.subplot_position = [0.1, 0.3, 0.85, 0.65]
        self.subplot_text_position = [0.1, 0.05, 0.85, 0.12]
        self.subplot_x_label = 'Time (s)'
        self.subplot_axis = None
        self.text_axis = None
        self.data_line_subplot = None
        self.text_box_subplot = None

        if default_subplot_creation:
            self.make_subplots()

    def make_subplots(self):
        self.create_plot_axis()
        self.create_text_box()

    def return_self_values(self):
        return self.subplot_axis, self.data_line_subplot, self.text_axis, self.text_box_subplot

    def create_plot_axis(self):
        # set axis for graph
        self.subplot_axis = self.figure.add_subplot(2, 1, 1)
        self.subplot_axis.set_position(self.subplot_position)
        self.data_line_subplot, = self.subplot_axis.plot([], [])
        plt.xlabel(self.subplot_x_label)
        plt.ylabel(self.reading_what_string + ' (' + self.reading_units_string + ')')

    def create_text_box(self):
        # set axis for text
        self.text_axis = self.figure.add_subplot(2, 1, 2)
        self.text_axis.set_position(self.subplot_text_position)
        plt.xticks([])
        plt.yticks([])

        # make text axis pretty
        self.text_axis.set_facecolor('lightsteelblue')
        self.text_box_subplot = self.text_axis.text(self.x_text, self.y_text, self.reading_what_string + ' = ',
                                                    transform=self.text_axis.transAxes, horizontalalignment='left',
                                                    verticalalignment='center', fontsize=24)
