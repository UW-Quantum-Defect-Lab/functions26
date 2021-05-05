# 2020-09-16
# This code was made for use in the Fu lab
# by Christian Zimmermann

import numpy as np
import pandas as pd
import scipy.signal as sps
from .useful_functions import calculate_weighted_mean


# Define class for finding peaks in data with local maxima
class Peaks:
    def __init__(self, data):
        # data should be a DataFrame containing columns 'x' and 'y' or a 2D array/list with:
        # array/list[0] = x_data and array/list[1] = y_data

        # Add parameters to self
        self.peaks = None
        self.data = None
        self.add_data(data)

    # Define function to add data
    def add_data(self, data):
        # Make sure data is stored as a dataframe with columns 'x' and 'y'
        if isinstance(data, pd.DataFrame):
            if all(data.columns == ['x', 'y']):
                self.data = data
            else:
                raise KeyError('Columns of data must be x and y.')
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data={'x': data[0], 'y': data[1]})
        else:
            raise TypeError('data must be a 2D array/list or a dataframe with columns x and y.')

        return True

    # Define function for finding peaks
    def find_peaks(self, prominence=1, number_of_data_points=2):
        # Prominence is used in scipy.signal's routines to identify local maxima present in data
        # Number of data points used for analysing peaks around local maxima

        # Find local maxima (LM)
        LM_indexes, _ = sps.find_peaks(self.data['y'].values, prominence=prominence)

        # Use weighted mean of data around LM_index to calculate true peak position
        peak_positions = []
        for LM_index in LM_indexes:
            data_LM = self.data.loc[LM_index - number_of_data_points: LM_index + number_of_data_points, :]
            data_LM = data_LM.nlargest(number_of_data_points, columns='y')
            peak_positions.append(calculate_weighted_mean(data_LM.x, data_LM.y))

        peaks = pd.DataFrame(data={'x': peak_positions, 'y': self.data.loc[LM_indexes, 'y']})
        peaks.reset_index(inplace = True, drop = True)
        self.peaks = peaks

        return True
