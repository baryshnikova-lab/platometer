"""Additional Platometer helper functions to open, save and plot data.
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.colors import LinearSegmentedColormap


def save_to_p(data, output_file=None):
    """Saves object to a pickle file.

    Args:
        data (obj): Object to be saved.
        output_file (str, optional): Destination path for saving.
    """

    if not output_file:
        output_file = os.path.join(os.getcwd(), 'output.p')

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle)


def load(file_path, version=3, verbose=True):
    """
    Loads all objects from a pickle or a HDF5 file.

    Args:
        file_path (str): Path to the input file.
        version (int, optional): Python version (2 or 3) in which the input file was saved.
        verbose (bool, optional): If True, show details.

    Returns:
        obj: The contents of the input file.
    """


    [_, file_extension] = os.path.splitext(file_path)

    if file_extension == '.h5':

        output = {}

        with pd.HDFStore(file_path) as store:
            fkeys = store.keys()

        for k in fkeys:
            k = k.lstrip(r'\/')
            if verbose:
                print(k)
            output[k] = pd.read_hdf(file_path, k)

    elif file_extension == '.p':

        if version == 3:
            output = pd.read_pickle(file_path)
        else:
            pickle_file = open(file_path, 'r')
            output = pickle.load(pickle_file)

        if verbose:
            print(', '.join(output.keys()))

    else:

        output = {}
        print("Extension unknown. Only pickle (.p) and HDF5 (.h5) files are supported.")

    return output


def plot_plate(data, colorbar=False, **kwargs):
    """Plots plate as a heatmap of colony sizes.

    Args:
        data (pandas.DataFrame): A DataFrame containing the quantified colony size data.
        colorbar (bool, optional): If True, plots the colorbar.
        **kwargs: Additional keyword arguments.

    Returns:
        matplotlib.axes.Axes containing the plate plot.
    """

    plate = np.zeros((32, 48)) + np.nan

    rows = data['row'].values.astype(int)
    cols = data['col'].values.astype(int)
    vals = data['size'].values.astype(float)

    plate[rows - 1, cols - 1] = vals

    if 'axes' in kwargs:
        axes = kwargs['axes']
    else:
        _, axes = plt.subplots(1, 1, figsize=(20, 10))

    vmin = kwargs.get('vmin', np.nanpercentile(vals, 5))
    vmax = kwargs.get('vmax', np.nanpercentile(vals, 95))
    midrange = kwargs.get('midrange',np.percentile(vals[(vals >= vmin) & (vals <= vmax)], [40, 60]))

    if 'ticklabels' in kwargs:
        xticklabels = kwargs['ticklabels']
        yticklabels = kwargs['ticklabels']
    else:
        xticklabels = False
        yticklabels = False

    img = axes.imshow(plate, cmap=red_green(),
                   norm=MidpointRangeNormalize(midrange=midrange),
                   interpolation='nearest',
                   vmin=vmin, vmax=vmax)

    axes.set_aspect('equal')
    axes.grid(False)

    if ~xticklabels:
        axes.set_xticks([])

    if ~yticklabels:
        axes.set_yticks([])

    if colorbar:
        plt.colorbar(img, ax=axes)

    plt.tight_layout()

    return axes


def red_green():
    """Creates a divergent colormap centered on black and ranging from red (low) to green (high).

    Returns:
        LinearSegmentedColormap in the red-black-green range.
    """

    color_dict = {'red': ((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
                  'green': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
                  'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
                  }

    cmap = LinearSegmentedColormap('RedGreen', color_dict)
    cmap.set_bad('gray', 0.5)

    return cmap


class MidpointRangeNormalize(colors.Normalize):
    """Normalizes colors to match a specified mid-range.
    """

    def __init__(self, vmin=None, vmax=None, midrange=None, clip=False):
        self.midrange = midrange
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x_values = [self.vmin, self.midrange[0], self.midrange[1], self.vmax]
        y_values = [0, 0.5, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x_values, y_values))
