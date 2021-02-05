# 2021-01-27
# This code was made for use in the Fu lab
# by Christian Zimmermann


from .constants import conversion_factor_nm_to_ev  # eV*nm
from .DataDictXXX import DataDictFilenameInfo
from .DataFrame26 import DataFrame26
from .Dict26 import Dict26
from .units import unit_families
from .useful_functions import get_added_label_from_unit
from matplotlib.colors import LogNorm, Normalize
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import scipy.io as spio
import warnings


def convert_to_string_to_float(number_string):
    # Figure out sign
    if 'n' in number_string:
        sign = -1
        number_string = number_string.split('n')[1]
    elif 'm' in number_string:
        sign = -1
        number_string = number_string.split('m')[1]
    elif '-' in number_string:
        sign = -1
        number_string = number_string.split('-')[1]
    else:
        sign = 1

    # Figure out decimal point
    if 'p' in number_string:
        number = float(number_string.replace('p', '.'))
    else:
        number = float(number_string)

    # Apply sign
    number *= sign

    return number


class DataImage:

    def __init__(self, file_name, folder_name, allowed_file_extensions):
        self.file_name = file_name
        if file_name == '':
            raise RuntimeError('File name is an empty string')
        self.folder_name = folder_name
        self.file_extension = self.file_name.split('.')[-1]
        self.check_file_type(allowed_file_extensions)

        self.file_info = DataDictFilenameInfo()
        self.get_file_info()

        self.image_data = np.zeros((1, 1))
        self.extent = {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
        self.get_data()

    def get_data(self):
        warnings.warn('Define your own get_data() function')
        pass

    def get_file_info(self):

        # Save filename without folder and file extension
        file_info_raw = self.file_name.split('.')[-2]
        if '/' in self.file_name:
            file_info_raw = file_info_raw.split('/')[-1]

        file_info_raw_components = file_info_raw.split('_')  # All file info are separated by '_'
        self.file_info.get_info(file_info_raw_components)  # retrieve info from file

        return True

    def check_file_type(self, allowed_file_extensions):
        allowed_file_extensions = [fe.lower() for fe in allowed_file_extensions]
        if self.file_extension.lower() not in allowed_file_extensions:
            raise RuntimeError('Given file extension does not much the allowed extensions: '
                               + str(allowed_file_extensions))


class DataConfocalScan(DataImage):
    allowed_file_extensions = ['mat']

    def __init__(self, file_name, folder_name='.', spectral_range='all', unit_spectral_range=None):

        self.spectral_range = spectral_range
        self.unit_spectral_range = unit_spectral_range
        super().__init__(file_name, folder_name, self.allowed_file_extensions)

    def get_data(self):
        def convert_xy_to_position_string(position):
            axes = ['x', 'y']
            string = ''
            for n, axis in enumerate(axes):
                string += axis
                if position[n] < 0:
                    position[n] = np.abs(position[n])
                    sign_string = 'n'
                else:
                    sign_string = ''
                string += sign_string
                string += str(np.round(position[n], 3)).replace('.', 'p')
                if n == 0:
                    string += '_'

            return string

        matlab_file_data = spio.loadmat(self.file_name)

        if 'scan' in matlab_file_data.keys():
            self.software = 'DoritoScopeConfocal'
            self.image_data = matlab_file_data['scan'][0][0][4]
            # Convert image, so it looks like what we see in the matlab GUI
            self.image_data = np.transpose(self.image_data)
            self.image_data = np.flip(self.image_data, axis = 0)

            self.x = matlab_file_data['scan'][0][0][0][0]
            self.y = matlab_file_data['scan'][0][0][1][0]
            self.y = np.flip(self.y)
        elif 'data' in matlab_file_data.keys():
            self.software = 'McDiamond'
            self.image_data = matlab_file_data['data'][0][0][7][0][0]
            # Convert image, so it looks like what we see in the matlab GUI
            self.image_data = np.transpose(self.image_data)
            self.image_data = np.flip(self.image_data, axis = 0)

            self.x = matlab_file_data['data'][0][0][2][0][0][0]
            self.y = matlab_file_data['data'][0][0][2][0][1][0]

        self.extent['x_min'] = np.min(self.x)
        self.extent['x_max'] = np.max(self.x)
        self.extent['y_min'] = np.min(self.y)
        self.extent['y_max'] = np.max(self.y)

        # Shift all values by half a pixel to have x, y position be associated with pixel center
        x_pixel_length = (self.extent['x_max'] - self.extent['x_min']) / (len(self.x) - 1)
        self.extent['x_min'] = self.extent['x_min'] - x_pixel_length / 2
        self.extent['x_max'] = self.extent['x_max'] + x_pixel_length / 2
        y_pixel_length = (self.extent['y_max'] - self.extent['y_min']) / (len(self.y) - 1)
        self.extent['y_min'] = self.extent['y_min'] - y_pixel_length / 2
        self.extent['y_max'] = self.extent['y_max'] + y_pixel_length / 2

        # Check whether spectrometer was used for data collection, if yes also import spectra
        if self.software == 'DoritoScopeConfocal':
            if matlab_file_data['scan'][0][0][3][0] == 'Spectrometer':
                self.type = 'Spectrometer'
                self.exposure_time = matlab_file_data['scan'][0][0][11][0][0]
                self.cycles = matlab_file_data['scan'][0][0][10][0][0]
                spectra_raw = matlab_file_data['scan'][0][0][15]
                self.spectra_raw = [[spectra_raw[ix][iy][0] for iy in range(len(self.y))] for ix in range(len(self.x))]
                self.spectra_raw = np.transpose(self.spectra_raw, axes=[1, 0, 2])
                self.spectra_raw = np.flip(self.spectra_raw, axis=0)
                self.spectra_raw = self.spectra_raw
                self.wavelength = matlab_file_data['scan'][0][0][16][0]
                self.wavelength = self.wavelength / 2
                self.photon_energy = conversion_factor_nm_to_ev / self.wavelength
                self.spectra = {}
                for ix, x_position in enumerate(self.x):
                    for iy, y_position in enumerate(self.y):
                        position_string = convert_xy_to_position_string([x_position, y_position])
                        self.spectra[position_string] = pd.DataFrame(
                            data={'x_nm': self.wavelength, 'y_counts_per_seconds': self.spectra_raw[iy][ix]})
                        self.spectra[position_string]['x_eV'] = self.photon_energy

                self.image_data_from_spectra = []
                for ix, x_position in enumerate(self.x):
                    integrated_counts_along_y = []
                    for iy, y_position in enumerate(self.y):
                        spectrum = pd.DataFrame(
                            data={'x_nm': self.wavelength, 'y_counts_per_seconds': self.spectra_raw[iy][ix]})
                        spectrum['x_eV'] = self.photon_energy
                        if self.spectral_range != 'all':
                            spectrum = spectrum.loc[
                                (spectrum['x_{0}'.format(self.unit_spectral_range)] >= self.spectral_range[0]) & (
                                            spectrum['x_{0}'.format(self.unit_spectral_range)] <= self.spectral_range[1])]
                        # Background correction
                        integrated_counts = spectrum['y_counts_per_seconds'].sum()
                        integrated_counts_along_y.append(integrated_counts)
                    self.image_data_from_spectra.append(integrated_counts_along_y)
                self.image_data_from_spectra = np.transpose(self.image_data_from_spectra)
            else:
                self.type = 'SPCM'
        else:
            self.type = 'SPCM'

        return True

    def add_image(self, scale_bar=None, scale='Auto', color_bar=True, plot_style=None, image_from_spectra=False,
                  masking_treshold=None, interpolation=None):
        # Define function for formatting scale bar text
        def format_scale_bar(scale_value, unit='um'):
            if unit == 'um':
                scale_string = r'{0}'.format(int(scale_value)) + r' $\mathrm{\mu}$m'
            return scale_string

        # Set plotting style
        if plot_style is not None:
            plt.style.use(plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

        # Plot image
        if image_from_spectra:
            self.image_data_to_plot = self.image_data_from_spectra
        else:
            self.image_data_to_plot = self.image_data
        if masking_treshold is not None:
            self.image_data_to_plot = np.ma.masked_where(self.image_data_to_plot <= masking_treshold,
                                                         self.image_data_to_plot)
        if scale == 'Auto':
            im = axes.imshow(self.image_data_to_plot,
                             cmap=plt.get_cmap('gray'),
                             extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                     self.extent['y_max']],
                             interpolation=interpolation)
        elif scale == 'Normalized':
            im = axes.imshow((self.image_data_to_plot - np.min(self.image_data_to_plot)) / (
                        np.max(self.image_data_to_plot) - np.min(self.image_data_to_plot)),
                             cmap=plt.get_cmap('gray'),
                             extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                     self.extent['y_max']],
                             interpolation=interpolation)
        else:
            default_scale = {'minimum_value': 0, 'maximum_value': np.Inf, 'norm': None, 'color_map': 'gray'}
            for key in default_scale:
                if key not in scale:
                    scale[key] = default_scale[key]

            if scale['norm'] is None:
                im = axes.imshow(self.image_data_to_plot,
                                 cmap=plt.get_cmap(scale['color_map']),
                                 extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                         self.extent['y_max']],
                                 vmin=scale['minimum_value'], vmax=scale['maximum_value'],
                                 interpolation=interpolation)
            elif scale['norm'] == 'log':
                im = axes.imshow(self.image_data_to_plot,
                                 cmap=plt.get_cmap(scale['color_map']),
                                 extent=[self.extent['x_min'], self.extent['x_max'], self.extent['y_min'],
                                         self.extent['y_max']],
                                 norm=LogNorm(vmin=scale['minimum_value'], vmax=scale['maximum_value']),
                                 interpolation=interpolation)

        # Add scale bar
        if scale_bar is not None:
            default_scale_bar = {'scale': 5, 'font_size': 24, 'color': 'white', 'position': 'lower left'}
            if not isinstance(scale_bar, dict):
                default_scale_bar['scale'] = scale_bar
                scale_bar = default_scale_bar
            else:
                for key in default_scale_bar:
                    if key not in scale_bar:
                        scale_bar[key] = default_scale_bar[key]
            fontprops = fm.FontProperties(size=scale_bar['font_size'])
            scalebar = AnchoredSizeBar(axes.transData,
                                       scale_bar['scale'],
                                       format_scale_bar(scale_bar['scale']),
                                       scale_bar['position'],
                                       pad=0.1,
                                       color=scale_bar['color'],
                                       frameon=False,
                                       size_vertical=0.5,
                                       sep=5,
                                       fontproperties=fontprops)

            axes.add_artist(scalebar)

        # Turn axes labels/ticks off
        plt.axis('off')

        # Display color bar
        if color_bar:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = figure.colorbar(im, cax=cax, orientation='vertical')
            if scale == 'Normalized':
                cax.set_ylabel('Normalized PL-Intensity (rel. units)')
            else:
                cax.set_ylabel('PL-Intensity (counts/second)')

        # Add figure and axes for to self further manipulation
        self.image = {'figure': figure, 'axes': axes}

        return True

    # Define function for adding markers on image
    def add_marker_to_image(self, marker=None):
        def convert_marker_string_to_marker_array(string):
            string_components = string.split('_')
            axes = ['x', 'y']
            marker = []
            for string_component in string_components:
                for axis in axes:
                    if axis in string_component:
                        number_string = string_component.split(axis)[1]
                        number = convert_to_string_to_float(number_string)
                        break
                marker.append(number)

            if len(marker) != 2:
                raise ValueError('String entered for marker is not in correct format.')

            return marker

        marker_default = {'type': 'dot', 'color': 'red', 'x_position': 0, 'y_position': 0, 'width': 0, 'height': 0,
                          'size': 100, 'position_string': None}
        # Test if self.image exists
        try:
            if not isinstance(marker, dict):
                if isinstance(marker, str):
                    marker = convert_marker_string_to_marker_array(marker)
                marker = {'x_position': marker[0], 'y_position': marker[1]}
            for key in marker_default:
                if key not in marker:
                    marker[key] = marker_default[key]

            # Overwrite x and y position with position_string if it was given
            if marker['position_string'] is not None:
                x_position, y_position = convert_marker_string_to_marker_array(marker['position_string'])
                marker['x_position'] = x_position
                marker['y_position'] = y_position

            # Add marker point
            if marker['type'] == 'dot':
                self.image['axes'].scatter(marker['x_position'], marker['y_position'], edgecolors='face',
                                           c=marker['color'], s=marker['size'])
            elif marker['type'] == 'area':
                if marker['size'] == 100:
                    marker['size'] = 3
                area = patches.Rectangle((marker['x_position'], marker['y_position']), marker['width'],
                                         marker['height'],
                                         linewidth=marker['size'], edgecolor=marker['color'], facecolor='none')
                self.image['axes'].add_patch(area)
        except NameError:
            raise NameError('Image was not added yet. Run .add_image() first!')

        return True

    def save_image(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.image['figure'].savefig(title, bbox_inches='tight', transparent=True)

        return True


class DataHyperSpectral:
    def __init__(self, file_name, spectral_ranges, unit_spectral_range='eV', folder_name='.'):

        self.spectral_ranges = spectral_ranges
        self.unit_spectral_range = unit_spectral_range
        self.file_name = file_name
        self.folder_name = folder_name
        self.spectral_scans = {}
        for n, spectral_range in enumerate(self.spectral_ranges):
            self.spectral_scans[n] = DataConfocalScan(self.file_name, self.folder_name, spectral_range,
                                                      self.unit_spectral_range)

    def add_hyperspectral_image(self, masking_treshold, scale_bar=None, plot_style=None, interpolation=None):
        def get_mask(image_data, n):
            mask = {}
            for k in image_data:
                if k != n:
                    mask[k] = image_data[n] > image_data[k]

            sum_mask = np.zeros(image_data[n].shape)
            for k in mask:
                sum_mask += mask[k]

            final_mask = np.zeros(image_data[n].shape)
            for i in range(len(sum_mask)):
                for j in range(len(sum_mask[i])):
                    if sum_mask[i][j] < len(image_data) - 1:
                        final_mask[i][j] = 0
                    else:
                        final_mask[i][j] = 1

            return final_mask

        # Define function for formatting scale bar text
        def format_scale_bar(scale_value, unit='um'):
            if unit == 'um':
                scale_string = r'{0}'.format(int(scale_value)) + r' $\mathrm{\mu}$m'
            return scale_string

        # Set color maps
        color_maps = ['Reds_r', 'Greens_r', 'Blues_r', 'Purples_r']

        # Set plotting style
        if plot_style != None:
            plt.style.use(plot_style)

        # Generate figure, axes
        figure = plt.figure(figsize=(15, 10))
        axes = figure.add_subplot(1, 1, 1)

        # Normalize image data
        self.image_data_to_plot = {}
        for n in range(len(self.spectral_ranges)):
            self.image_data_to_plot[n] = (self.spectral_scans[n].image_data_from_spectra - np.min(
                self.spectral_scans[n].image_data_from_spectra))
            self.image_data_to_plot[n] = self.image_data_to_plot[n] / np.max(self.image_data_to_plot[n])

        # Mask and plot image data
        for n in range(len(self.spectral_ranges)):
            if masking_treshold != None:
                mask = self.image_data_to_plot[n] <= masking_treshold
            else:
                mask = get_mask(self.image_data_to_plot, n)
            self.image_data_to_plot[n] = np.ma.masked_array(self.image_data_to_plot[n], mask)
            im = axes.imshow(self.image_data_to_plot[n],
                             cmap=plt.get_cmap(color_maps[n]),
                             extent=[self.spectral_scans[n].extent['x_min'], self.spectral_scans[n].extent['x_max'],
                                     self.spectral_scans[n].extent['y_min'], self.spectral_scans[n].extent['y_max']],
                             interpolation=interpolation)

        # Add scale bar
        if scale_bar is not None:
            default_scale_bar = {'scale': 5, 'font_size': 24, 'color': 'white', 'position': 'lower left'}
            if not isinstance(scale_bar, dict):
                default_scale_bar['scale'] = scale_bar
                scale_bar = default_scale_bar
            else:
                for key in default_scale_bar:
                    if key not in scale_bar:
                        scale_bar[key] = default_scale_bar[key]
            fontprops = fm.FontProperties(size=scale_bar['font_size'])
            scalebar = AnchoredSizeBar(axes.transData,
                                       scale_bar['scale'],
                                       format_scale_bar(scale_bar['scale']),
                                       scale_bar['position'],
                                       pad=0.1,
                                       color=scale_bar['color'],
                                       frameon=False,
                                       size_vertical=0.5,
                                       sep=5,
                                       fontproperties=fontprops)

            axes.add_artist(scalebar)

        # Turn axes labels/ticks off
        plt.axis('off')

        # Add figure and axes for to self further manipulation
        self.image = {'figure': figure, 'axes': axes}

        return True

    def save_image(self, title, file_extension='pdf', folder='.'):
        title = '{0}/{1}.{2}'.format(folder, title, file_extension)
        self.image['figure'].savefig(title, bbox_inches='tight', transparent=True, facecolor='black')

        return True