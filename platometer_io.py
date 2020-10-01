""" code to open, save and plot data from platometer.py"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.colors import LinearSegmentedColormap

def save_to_p(data, output_file=None):
    """save data as pickle file"""
    if not output_file:
        output_file = os.path.join(os.getcwd(), 'output.p')

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle)


def load(file_path, version=3, verbose=True):

    """
    # Loads all objects from a pickle or a HDF5 file
    :param file_path: local path to the input file
    :param v: python version
    :param verbose:
    :return: contents of the input file
    """

    # home = expanduser('~')
    # file_path = re.sub('~', home, file_path)

    [_, file_extension] = os.path.splitext(file_path)

    if file_extension == '.h5':

        output = {}

        with pd.HDFStore(file_path) as store:
            fkeys = store.keys()

        for k in fkeys:
            k = k.lstrip('\/')
            if verbose:
                print(k)
            output[k] = pd.read_hdf(file_path, k)

        return output

    elif file_extension == '.p':

        if version == 3:
            output = pd.read_pickle(file_path)
        else:
            pickle_file = open(file_path, 'r')
            output = pickle.load(pickle_file)

        if verbose:
            print(', '.join(output.keys()))
        return output

    else:
        print("Extension unknown. Only pickle (.p) and HDF5 (.h5) files are supported.")


def plot_plate(data, colorbar=False, **kwargs):
    """plot colony quantification of plates with colonies color coded by size"""
    plate = np.zeros((32, 48)) + np.nan

    rows = data['row'].values.astype(int)
    cols = data['col'].values.astype(int)
    vals = data['size'].values.astype(float)

    plate[rows - 1, cols - 1] = vals

    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    vmin = kwargs.get('vmin',np.nanpercentile(vals, 5))
    vmax = kwargs.get('vmax',np.nanpercentile(vals, 95))
    midrange = kwargs.get('midrange',np.percentile(vals[(vals >= vmin) & (vals <= vmax)], [40, 60]))

    if 'ticklabels' in kwargs:
        xticklabels = kwargs['ticklabels']
        yticklabels = kwargs['ticklabels']
    else:
        xticklabels = False
        yticklabels = False

    img = ax.imshow(plate, cmap=red_green(),
                   norm=MidpointRangeNormalize(midrange=midrange),
                   interpolation='nearest',
                   vmin=vmin, vmax=vmax)

    ax.set_aspect('equal')
    #     ax.set_xlim(-1, 48)
    #     ax.set_ylim(-1, 32)
    #     ax.invert_yaxis()
    ax.grid(False)

    if ~xticklabels:
        ax.set_xticks([])

    if ~yticklabels:
        ax.set_yticks([])

    if colorbar:
        plt.colorbar(img, ax=ax)

    plt.tight_layout()

    return img, ax


def red_green():
    """create colormap with range red (low) to green (high)"""
    color_dict = {'red': ((0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
                  'green': ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
                  'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
                  }

    cmap = LinearSegmentedColormap('RedGreen', color_dict)
    cmap.set_bad('gray', 0.5)

    return cmap


class MidpointRangeNormalize(colors.Normalize):
    """normalize colors by data midpoints"""
    def __init__(self, vmin=None, vmax=None, midrange=None, clip=False):
        self.midrange = midrange
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x_values = [self.vmin, self.midrange[0], self.midrange[1], self.vmax]
        y_values = [0, 0.5, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x_values, y_values))
