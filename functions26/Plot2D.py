# 2021-05-01
# This code was made for use in the Fu lab
# by Christian Zimmermann

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


class two_dimensional_plot:
    def __init__(self, data,
                 x_axis, y_axis,
                 x_axis_label='', y_axis_label='',
                 axes_limits='Auto',
                 scale='Auto',
                 color_bar=True, color_bar_label='',
                 shading='auto',
                 plot_style=None):

        self.data = data
        self.scale = scale
        self.color_bar = color_bar
        self.color_bar_label = color_bar_label
        self.plot_style = plot_style
        self.shading = shading

        self.x_axis = x_axis
        self.y_axis = y_axis

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

        self.axes_limits = axes_limits

        self.add_plot()

    def add_plot(self):
        # Set plotting style
        if self.plot_style is not None:
            plt.style.use(self.plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

        # Axes
        default_axes_limits = {'x': [np.nanmin(self.x_axis), np.nanmax(self.x_axis)],
                               'y': [np.nanmin(self.y_axis), np.nanmax(self.y_axis)]}
        if self.axes_limits == 'Auto':
            self.axes_limits = default_axes_limits
        else:
            for key in default_axes_limits:
                if key not in self.axes_limits:
                    self.axes_limits[key] = default_axes_limits[key]

        if self.scale == 'Auto':
            im = plt.pcolormesh(self.x_axis, self.y_axis, self.data,
                                cmap=plt.get_cmap('gray'),
                                shading=self.shading,
                                rasterized=True
                                )

        else:
            default_scale = {'minimum_value': np.nanmin(self.data), 'maximum_value': np.nanmax(self.data), 'norm': None,
                             'color_map': 'gray'}
            for key in default_scale:
                if key not in self.scale:
                    self.scale[key] = default_scale[key]

            if self.scale['norm'] is None:
                im = plt.pcolormesh(self.x_axis, self.y_axis, self.data,
                                    cmap=self.scale['color_map'],
                                    vmin=self.scale['minimum_value'], vmax=self.scale['maximum_value'],
                                    shading=self.shading,
                                    rasterized=True
                                    )
            elif self.scale['norm'] == 'log':
                im = plt.pcolormesh(self.x_axis, self.y_axis, self.data,
                                    cmap=self.scale['color_map'],
                                    norm=LogNorm(vmin=self.scale['minimum_value'], vmax=self.scale['maximum_value']),
                                    shading=self.shading,
                                    rasterized=True
                                    )

        if self.color_bar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = figure.colorbar(im, cax=cax, orientation='vertical')
            cax.set_ylabel(self.color_bar_label)

        # Set axes limits
        axes.set_xlim(self.axes_limits['x'])
        axes.set_ylim(self.axes_limits['y'])

        # Set axes labels
        axes.set_xlabel(self.x_axis_label)
        axes.set_ylabel(self.y_axis_label)

        # Add figure and axes for to self further manipulation
        self.plot = {'figure': figure, 'axes': axes}

        return True

    def save_plot(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.plot['figure'].savefig(title, bbox_inches='tight', transparent=True)

        return True
