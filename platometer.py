"""Platometer is a simple image-processing tool for quantifying colony sizes
from arrayed growth experiments in yeast or bacteria.
Typical applications include phenotypic screens of mutant collections
and synthetic genetic array (SGA) experiments.

Authors: Gina Turco, Anastasia Baryshnikova (2020)
"""

import datetime
import time
import os
import argparse
import pickle

from os.path import expanduser
import multiprocessing as mp
import numpy as np
import pandas as pd

from scipy.signal import medfilt2d
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from skimage import filters
from skimage.morphology import watershed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from platometer_io import load, save_to_p, plot_plate
from platometer_utils import detect_peaks, bucket, fit_sin


class Platometer:
    """ Main class that stores input data, intermediate processing steps
    and, ultimately, colony size measurements.
    """

    def __init__(self, path_to_image_file, plate_format=np.array([32, 48]), verbose=True):

        self.path_to_image_file = path_to_image_file
        self.verbose = verbose
        self.image = plt.imread(path_to_image_file)
        self.n_rows, self.n_cols = plate_format

        self.im_center = None
        self.fit_row = None
        self.fit_col = None

        self.im_gray_trimmed = None
        self.im_foreground = None
        self.im_objects = None

        # List of colony center coordinates
        self.colony_pxl = None

        # DataFrame with final output
        self.colony_data = None

        self.row_avg_fit = None
        self.col_avg_fit = None

    def get_path_to_output_file(self, path='', extension=''):
        """Returns the absolute path to the output file.

        Args:
            path (str): User-provided path to the output file.
            extension (str): Desired extension for the output file
                (which will be located in the same directory as the original image)

        Returns:
            str: Path to the output file.
        """

        if path:
            path_to_output_file = path.replace('~', expanduser('~'))
        else:
            path_to_output_file = self.path_to_image_file + extension

        already_exists = os.path.isfile(path_to_output_file)
        if already_exists:
            raise FileExistsError(('File %s already exists. '
                                   'Please specify a different path '
                                   'or a different extension.' % path_to_output_file))

        return path_to_output_file

    def save(self, path='', verbose=False):
        """Saves this Platometer instance into a pickle file.

        Args:
            path (str): Path to the output file.
            verbose (bool): If True, shows details.
        """

        path_to_p_file = self.get_path_to_output_file(
            path=path, extension='.p')

        if verbose:
            print('Saving to %s.' % path_to_p_file)

        # (Hacky) Remove the fit_sin lambda to allow pickle to be used
        fit_row = self.fit_row
        fit_col = self.fit_col

        del self.fit_row['fitfunc']
        del self.fit_col['fitfunc']

        with open(path_to_p_file, 'wb') as file_handle:
            pickle.dump(self, file_handle)

        self.fit_row = fit_row
        self.fit_col = fit_col

    def print(self, path='', verbose=False):
        """Prints the quantified colony size data into a text file.

        Args:
            path (str): Path to the output file.
            verbose (bool): If True, shows details.
        """

        path_to_dat_file = self.get_path_to_output_file(
            path=path, extension='.dat')

        if verbose:
            print('Saving to %s.' % path_to_dat_file)

        self.colony_data['circularity'] = np.nan
        self.colony_data['flags'] = np.nan

        outfile = open(path_to_dat_file, 'w')
        outfile.write(format('# Data file generated by platometer on %s\n' %
                             datetime.datetime.now()))
        outfile.write('# row\tcol\tsize\tcircularity\tflags\n')
        self.colony_data.to_csv(outfile, columns=['row', 'col', 'size', 'circularity', 'flags'],
                                sep='\t', header=False, index=False)
        outfile.close()

    def gray_and_trim(self):
        """"Converts image to gray scale and trims the plate edges.
        """

        # Convert to gray scale
        im_gray = np.nanmean(self.image, axis=2)

        # Downsample 2X
        im_gray = bucket(im_gray, [2, 2])

        # Medial filter
        im_gray = medfilt2d(im_gray, kernel_size=5)

        # Get row & col averages
        row_avg = np.nanmean(im_gray, axis=1)
        col_avg = np.nanmean(im_gray, axis=0)

        row_pxl = np.arange(len(row_avg))
        col_pxl = np.arange(len(col_avg))

        # Middle 50% of the plate
        row_pxl_mid = np.percentile(row_pxl, [25, 75]).astype(int)
        col_pxl_mid = np.percentile(col_pxl, [25, 75]).astype(int)

        row_pxl_mid_range = np.arange(row_pxl_mid[0], row_pxl_mid[1])
        col_pxl_mid_range = np.arange(col_pxl_mid[0], col_pxl_mid[1])

        # Find the positions of colony centers by fitting a sinusoid to the row and col averages
        # First fit (only on the middle of the plate, to get robust estimates of parameters)
        fit_row_0 = fit_sin(row_pxl_mid_range, row_avg[row_pxl_mid_range])
        fit_col_0 = fit_sin(col_pxl_mid_range, col_avg[col_pxl_mid_range])

        # Second fit (with the guesses from the first fit)
        self.fit_row = fit_sin(row_pxl, row_avg, guess=fit_row_0['popt'])
        self.fit_col = fit_sin(col_pxl, col_avg, guess=fit_col_0['popt'])

        row_avg_fit = self.fit_row['fitfunc'](row_pxl)
        col_avg_fit = self.fit_col['fitfunc'](col_pxl)

        # Define the expected size of a window containing the expected number of rows or columns
        w_row = np.ceil(self.fit_row['period'] * self.n_rows).astype(int)
        w_col = np.ceil(self.fit_col['period'] * self.n_cols).astype(int)

        # Find the boundaries of the plate (when plate is not centered within the image)
        plate_row_boundaries = [np.min(np.nonzero(row_avg > 100)),
                                np.max(np.nonzero(row_avg > 100))]
        plate_col_boundaries = [np.min(np.nonzero(col_avg > 100)),
                                np.max(np.nonzero(col_avg > 100))]

        # Find the approximate middle of the plate
        best_w_row_mid_pxl = int(np.min(plate_row_boundaries) +
                                 np.abs(np.diff(plate_row_boundaries) / 2))
        best_w_col_mid_pxl = int(np.min(plate_col_boundaries) +
                                 np.abs(np.diff(plate_col_boundaries) / 2))

        # Snap to the closest negative peak of the fitted sinusoid
        row_negative_peaks_px = detect_peaks(row_avg_fit, mpd=self.fit_row['period'] / 2,
                                             edge='both', kpsh=True, valley=True, show=False)
        col_negative_peaks_px = detect_peaks(col_avg_fit, mpd=self.fit_col['period'] / 2,
                                             edge='both', kpsh=True, valley=True, show=False)

        best_w_row_mid_pxl = row_negative_peaks_px[np.argmin(np.abs(best_w_row_mid_pxl -
                                                                    row_negative_peaks_px))]
        best_w_col_mid_pxl = col_negative_peaks_px[np.argmin(np.abs(best_w_col_mid_pxl -
                                                                    col_negative_peaks_px))]

        trim_pxl_row = [best_w_row_mid_pxl - w_row / 2 - self.fit_row['period'] / 4,
                        best_w_row_mid_pxl + w_row / 2 + self.fit_row['period'] / 4]
        trim_pxl_col = [best_w_col_mid_pxl - w_col / 2 - self.fit_col['period'] / 4,
                        best_w_col_mid_pxl + w_col / 2 + self.fit_col['period'] / 4]

        # Trim the image around the best window
        im_gray_trimmed = im_gray.copy()

        row_mask = row_pxl[(row_pxl > trim_pxl_row[0]) &
                           (row_pxl < trim_pxl_row[1])]
        col_mask = col_pxl[(col_pxl > trim_pxl_col[0]) &
                           (col_pxl < trim_pxl_col[1])]
        im_gray_trimmed = im_gray_trimmed[np.ix_(row_mask, col_mask)]

        best_w_row_mid_pxl = best_w_row_mid_pxl - trim_pxl_row[0]
        best_w_col_mid_pxl = best_w_col_mid_pxl - trim_pxl_col[0]

        self.im_gray_trimmed = im_gray_trimmed
        self.im_center = [best_w_row_mid_pxl, best_w_col_mid_pxl]

        self.row_avg_fit = row_avg_fit[(
            row_pxl > trim_pxl_row[0]) & (row_pxl < trim_pxl_row[1])]
        self.col_avg_fit = col_avg_fit[(
            col_pxl > trim_pxl_col[0]) & (col_pxl < trim_pxl_col[1])]

    def detect_colonies(self):
        """Identifies the centers of each colony on the plate.
        """

        # --- Find the local maxima (= locations of colony centers)
        # Rows
        colony_row_pxl = detect_peaks(self.row_avg_fit, mpd=self.fit_row['period'] / 2,
                                      edge='both', kpsh=True, valley=False, show=False)

        # Columns
        colony_col_pxl = detect_peaks(self.col_avg_fit, mpd=self.fit_col['period'] / 2,
                                      edge='both', kpsh=True, valley=False, show=False)

        # --- Merge colonies that are very close
        [colony_col_pxl, _] = merge_colonies(colony_col_pxl)
        [colony_row_pxl, _] = merge_colonies(colony_row_pxl)

        # --- Only include the middle n_rows (just in case we detect more peaks than expected rows)
        mid_row = int(self.n_rows/2)
        colony_distance_from_center = self.im_center[0] - colony_row_pxl
        row_right_side = np.sort(
            -colony_distance_from_center[colony_distance_from_center < 0])[0:mid_row]
        row_left_side = np.sort(
            colony_distance_from_center[colony_distance_from_center > 0])[0:mid_row]
        colony_row_pxl = np.sort(self.im_center[0] - np.concatenate((row_left_side,
                                                            -row_right_side), axis=0)).astype(int)

        # --- Only include the middle 48 columns (just in case more than 48 peaks are detected)
        mid_col = int(self.n_cols/2)
        colony_distance_from_center = self.im_center[1] - colony_col_pxl
        col_right_side = np.sort(
            -colony_distance_from_center[colony_distance_from_center < 0])[0:mid_col]
        col_left_side = np.sort(
            colony_distance_from_center[colony_distance_from_center > 0])[0:mid_col]
        colony_col_pxl = np.sort(self.im_center[1] - np.concatenate((col_left_side,
                                                            -col_right_side), axis=0)).astype(int)

        [colony_col_pxl_2d, colony_row_pxl_2d] = np.meshgrid(
            colony_col_pxl, colony_row_pxl)
        self.colony_pxl = list(
            zip(colony_row_pxl_2d.ravel(), colony_col_pxl_2d.ravel()))

    def measure_colony_sizes(self):
        """Quantifies the size of each colony.
        """

        # Estimate foreground
        # im_foreground = estimate_foreground_by_gmm(im_gray_trimmed,
        #                                colony_row_pxl_2d, colony_col_pxl_2d, n_components=3)
        # im_foreground = estimate_foreground_by_otsu(im_gray_trimmed)
        self.im_foreground = estimate_foreground_by_adaptive_thresholding(self.im_gray_trimmed,
                                                            block_size=self.fit_row['period'] * 5)

        # Find all objects
        [im_objects, _] = ndi.label(self.im_foreground)

        [colony_row_pxl_2d, colony_col_pxl_2d] = zip(*self.colony_pxl)

        # Only keep objects that correspond to colonies
        colony_objects = im_objects[colony_row_pxl_2d, colony_col_pxl_2d]
        im_objects[~np.isin(im_objects, colony_objects)] = 0

        # Close the small gaps within objects
        im_objects = ndi.binary_closing(
            im_objects, structure=np.ones((2, 2)), iterations=1)

        # Repeat the labeling
        [im_objects, _] = ndi.label(im_objects)

        # Identify touching colonies (if any) and separate them using watershed segmentation

        # 1. Generate a matrix of colony center markers
        colony_center_coords = np.array(self.colony_pxl).T
        markers = np.zeros(im_objects.shape)
        markers[tuple(colony_center_coords)] = np.arange(
            colony_center_coords.shape[1])

        # 2. Calculate distance to the background for each pixel in each object
        im_foreground_objects = im_objects > 0
        distance = ndi.distance_transform_edt(im_foreground_objects)

        # (Hacky) Getting the negative distance (pylint raises an E1130 for -distance)
        distance = np.multiply(distance, -1)

        # 3. Label
        im_objects = watershed(distance, markers, mask=im_foreground_objects)

        self.im_objects = im_objects

        data = {'col_pxl': colony_col_pxl_2d,
                'row_pxl': colony_row_pxl_2d,
                'label': self.im_objects[colony_row_pxl_2d, colony_col_pxl_2d].astype(int)}

        colony_data = pd.DataFrame(data=data)

        colony_data['col'] = np.digitize(colony_data['col_pxl'].values,
                                         colony_data['col_pxl'].unique())
        colony_data['row'] = np.digitize(colony_data['row_pxl'].values,
                                         colony_data['row_pxl'].unique())

        # Measure colony size = number of pixels assigned to each colony object
        colony_objects = self.im_objects.ravel().astype(int)
        # ignore object 0 (= empty spots matching background)
        colony_objects = colony_objects[colony_objects > 0]
        colony_size = np.bincount(colony_objects)

        colony_data['size'] = colony_size[colony_data['label'].values]
        colony_data['size'] = colony_data['size'].astype(float)

        self.colony_data = colony_data

    def process(self):
        """Runs Platometer on an image.
        Args:
            image (dict): A dictionary containing the path to the image and other parameters.
        Returns:
            Platometer: The Platometer instance containing inputs and outputs of image processing.
        """

        try:
            self.gray_and_trim()
            self.detect_colonies()
            self.measure_colony_sizes()
        except RuntimeError:
            print('RuntimeError with processing %s. Moving on...' % image['path'])


    def get_colony_data(self):
        """Returns the quantified colony size data.
        """

        return self.colony_data

    def test(self):
        """Tests the output by checking the number of detected rows & columns
        (i.e., rows and cols with >50% of values).
        """

        data_test = self.colony_data.loc[pd.notnull(self.colony_data['size'])]
        n_rows = np.sum(data_test.groupby('row')[
                        'size'].count() > self.n_cols/2)
        n_cols = np.sum(data_test.groupby('col')[
                        'size'].count() > self.n_rows/2)

        n_rows_missing = self.n_rows-n_rows
        n_cols_missing = self.n_cols-n_cols

        if (n_rows_missing > 0) | (n_cols_missing > 0):
            if self.verbose:
                print("Warning: %d rows and %d columns are missing." %
                      (n_rows_missing, n_cols_missing))

        # # Check for giant colonies & mask them
        # md = np.nanmedian(self.colony_data['size'])
        # is_giant_colony = self.colony_data['size'] > md*10
        # if np.sum(is_giant_colony) > 0:
        #     if self.verbose:
        #         print(format('Warning: Masking %d giant colonies.' % np.sum(is_giant_colony)))
        # self.colony_data.loc[is_giant_colony, 'size'] = np.nan

    def show_plate(self, axes=None, **kwargs):
        """Plots the plate at various stages of processing.

        Args:
            axes (matplotlib.axes.Axes, optional): Axes handle to be used.
            **kwargs: Additional keyword arguments.

        Returns:
            matplotlib.axes.Axes: Axes containing the plate plot.
        """

        if 'show' in kwargs:
            img = getattr(self, kwargs['show'])
            if kwargs['show'] == 'im_objects':
                colors = np.vstack(
                    ([0, 0, 0], np.random.rand(self.n_rows*self.n_cols, 3)))
                cmap = matplotlib.colors.ListedColormap(colors)
                kwargs2 = {'cmap': cmap}
            elif kwargs['show'] == 'im_foreground':
                cmap = 'gnuplot'
                kwargs2 = {'cmap': cmap}
            elif kwargs['show'] == 'image':
                kwargs2 = {'cmap': 'gray', 'vmin': 350, 'vmax': 600}
            else:
                cmap = 'gray'
                kwargs2 = {'cmap': cmap}
        else:
            kwargs['show'] = 'plate'
            img = self.im_gray_trimmed
            kwargs2 = {'cmap': 'gray', 'vmin': 350, 'vmax': 600}

        if not axes:
            axes = plt.axes()

        if 'row' in kwargs:
            row = kwargs['row']
            col = kwargs['col']

            rows_selected = self.colony_data['row'].isin(
                [row * 2, row * 2 - 1])
            cols_selected = self.colony_data['col'].isin(
                [col * 2, col * 2 - 1])

            y_pxl = self.colony_data.loc[rows_selected &
                                         cols_selected, 'row_pxl'].max()
            x_pxl = self.colony_data.loc[rows_selected &
                                         cols_selected, 'col_pxl'].min()

            x_pxl = x_pxl - self.fit_col['period'] / 2
            y_pxl = y_pxl + self.fit_row['period'] / 2

            width = self.fit_col['period'] * 2
            height = -self.fit_row['period'] * 2

            x_pxl = np.round(x_pxl - width).astype(int)
            y_pxl = np.round(y_pxl - height).astype(int)

            width = np.round(width * 3).astype(int)
            height = np.round(height * 3).astype(int)

            axes.imshow(img[y_pxl + height:y_pxl,
                          x_pxl:x_pxl + width], **kwargs2)

        elif kwargs['show'] == 'colony_data':

            axes = plot_plate(img, axes=axes, **kwargs)

        else:

            axes.imshow(img, **kwargs2)

        axes.set_xticks([], [])
        axes.set_yticks([], [])

        return axes

    def show_position(self, row, col, axes=None, **kwargs):
        """Plots a section of a plate as defined by the specified row and column.

        Args:
            row (int): Row position (in 384 format) to be displayed.
            col (int): Column position (in 384 format) to be displayed.
            axes (matplotlib.axes.Axes, optional): Axes handle to be used.
            **kwargs: Additional keyword arguments.
        """

        e_color = kwargs.get('c', 'r')

        rows_selected = self.colony_data['row'].isin([row * 2, row * 2 - 1])
        cols_selected = self.colony_data['col'].isin([col * 2, col * 2 - 1])

        y_pxl = self.colony_data.loc[rows_selected &
                                     cols_selected, 'row_pxl'].max()
        x_pxl = self.colony_data.loc[rows_selected &
                                     cols_selected, 'col_pxl'].min()

        x_pxl = x_pxl - self.fit_col['period'] / 2
        y_pxl = y_pxl + self.fit_row['period'] / 2

        width = self.fit_col['period'] * 2
        height = -self.fit_row['period'] * 2

#         Bottom left x and y coordinates of the square
#         x_pxl = x_pxl-w
#         y_pxl = y_pxl-h

#         Width and height of the square
#         w = w*3
#         h = h*3

        rect = patches.Rectangle((x_pxl, y_pxl), width, height,
                                 linewidth=2, edgecolor=e_color, facecolor='none')

        if not axes:
            axes = plt.axes()

#         axes.imshow(self.im_gray_trimmed)

        axes.add_patch(rect)


def merge_colonies(pxl, vals=np.nan, distance_threshold=15):
    """Merges close colonies (assumed to be a detection artifact).

    Args:
        pxl (numpy array): Row or column positions of the colony centers.
        vals (float): Fill value.
        distance_threshold (int, optional): Minimum distance between distinct objects.

    Returns:
        numpy array: Row or column positions of the new colonies.
        numpy array: Foreground values of the new colonies.
    """

    distances = np.diff(pxl)
    labels = np.insert(np.cumsum(distances > distance_threshold), 0, 0)
    pxl_df = pd.DataFrame(data={'label': labels, 'pxl': pxl, 'val': vals})
    df_merged = pxl_df.groupby('label')[['pxl', 'val']].mean()

    return df_merged['pxl'].values.astype(int), df_merged['val'].values


def estimate_foreground_by_gmm(im_gray_trimmed, colony_row_pxl_2d,
                               colony_col_pxl_2d, n_components=3):
    """Estimates image background vs foreground using a 3-component Gaussian Mixed Model.

    Args:
        im_gray_trimmed (numpy array): Trimmed gray-scale image.
        colony_row_pxl_2d (numpy array):
            2-D matrix of y-coordinates (rows) of colony centers (from meshgrid).
        colony_col_pxl_2d (numpy array):
            2-D matrix of x-coordinates (columns) of colony centers (from meshgrid).
        n_components (int, optional): Number of components for the Gaussian Mixed Model.

    Returns:
        numpy array: Thresholded foreground values.
    """

    igt = im_gray_trimmed.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full').fit(igt)
    cluster_labels = gmm.predict(igt)

    im_cluster_labels = cluster_labels.reshape(im_gray_trimmed.shape)

    height = np.bincount(
        im_cluster_labels[colony_row_pxl_2d.ravel(), colony_col_pxl_2d.ravel()])
    colony_cluster = np.argmax(height)

    im_foreground = im_cluster_labels == colony_cluster

    return im_foreground


def estimate_foreground_by_otsu(im_gray_trimmed):
    """Estimates image background vs foreground using Otsu thresholding.

    Args:
        im_gray_trimmed (numpy array): Trimmed gray-scale image.

    Returns:
        numpy array: Thresholded foreground values.
    """

    threshold = filters.threshold_otsu(im_gray_trimmed)
    im_foreground = im_gray_trimmed > threshold

    return im_foreground


def estimate_foreground_by_adaptive_thresholding(im_gray_trimmed, block_size=45):
    """Estimates image background vs foreground using an adaptive threshold based
       on the local neighborhood mean.

    Args:
        im_gray_trimmed (numpy array): Trimmed gray-scale image.
        block_size (int, optional): Local window to consider when estimating background.

    Returns:
        numpy array: Thresholded foreground values.
    """

    # Round block size to the nearest odd integer
    block_size = int(np.ceil(block_size) // 2 * 2 + 1)

    adaptive_thresh = filters.threshold_local(im_gray_trimmed, block_size,
                                              mode='nearest', method='mean')

    adaptive_thresh = filters.gaussian(
        adaptive_thresh, sigma=10, mode='nearest')
    im_foreground = im_gray_trimmed > adaptive_thresh

    return im_foreground


def run_platometer(image, save_to_file=False, verbose=True):
    """Runs Platometer on an image.

    Args:
        image (dict): A dictionary containing the path to the image and other parameters.
        save_to_file (bool, optional): If True, save the Platometer instance to file.
        verbose (bool, optional): If True, show details.

    Returns:
        Platometer: The Platometer instance containing inputs and outputs of image processing.
    """

    image_path = image['path'].replace('~', expanduser('~'))
    plat = Platometer(
        image_path, plate_format=image['plate_format'], verbose=verbose)
    plat.process()
    # Saves the object to pickle
    if save_to_file:
        plat.save()

    return plat


def run_platometer_batch(image, verbose=False):
    """Runs Platometer on a batch of images. Used for multiprocessing.

    Args:
        image (dict): A dictionary containing the path to the image and other parameters.
        verbose (bool, optional): If True, show details.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed colony size data.
    """

    plate = run_platometer(image, verbose=verbose)
    plate.test()

    colony_data = plate.get_colony_data()

    if isinstance(colony_data, pd.DataFrame):
        cols = [c for c in image.keys() if c not in ['path', 'plate_format']]
        for col in cols:
            colony_data[col] = image[col]

    return colony_data

def process_folder(folder):
    """creates the processed data folder and returns a list of image files to process"""

    # Get path to the folder to contain processed data
    current_date = datetime.datetime.today().strftime('%Y%m%d')
    sub_folder = 'platometer_' + current_date
    data_folder = os.path.join(folder, sub_folder)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print('Created %s' % data_folder)

    # Get list of JPG files in the folder
    jpg_files = [os.path.join(folder, f) for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)
                                ) and f.lower().endswith('jpg')
                 and os.path.getsize(os.path.join(folder, f)) > 0]
    
    return(jpg_files, data_folder)


def process_and_save_chunk(n_chunk, ix_chunk_start, ix_chunk_stop, jpg_files,
        jpg_files_id, plate_format, nr_processes, data_folder, start):
    """runs platometer on current chunk and saves output to a temp file"""
    
    this_jpg_files_dict = [{'path': jpg_files[i], 'file_id': jpg_files_id[i],
                            'plate_format': np.array(plate_format)} for i in
                           np.arange(ix_chunk_start, ix_chunk_stop+1)]

    chunk_data = pd.DataFrame()

    if nr_processes > 1:
        pool = mp.Pool(processes=nr_processes)
        for res in pool.map_async(run_platometer_batch, this_jpg_files_dict).get():
            if res.shape[0] > 0:
                chunk_data = chunk_data.append(res, ignore_index=True)
    else:
        for im in this_jpg_files_dict:
            chunk_data = chunk_data.append(
                run_platometer_batch(im), ignore_index=True)

    # Temporarily save the chunk data
    path_to_this_chunk_file = os.path.join(
        data_folder, format('chunk%d_data.p' % n_chunk))
    save_to_p(chunk_data, path_to_this_chunk_file)

    print('Execution time: %.2f seconds' % (time.time()-start))
    start = time.time()



if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser(description='Process plates')
    parser.add_argument('path_to_image_folder_list', metavar='path_to_image_folder_list', type=str,
                        help='Path to the file containing the list of image folders to process')
    parser.add_argument('--plate_format', metavar='plate_format', type=int, nargs=2,
                        default=[32, 48], help='Expected plate format (default=32 48)')
    parser.add_argument('--chunk_size', metavar='chunk_size', type=int, default=100,
                        help='Number of images per chunk (default=100)')
    parser.add_argument('--nr_processes', metavar='nr_processes', type=int,
                        help='Number of cores to use (default: all available)')

    args = parser.parse_args()

    CHUNK_SIZE = 100
    if args.chunk_size:
        CHUNK_SIZE = args.chunk_size

    nr_processes = mp.cpu_count()
    if args.nr_processes:
        nr_processes = args.nr_processes

    folders = pd.read_csv(args.path_to_image_folder_list,
                          sep='\t', header=None)

    for folder in folders[0]:

        folder = folder.replace('~', expanduser('~'))
        print('Processing %s' % folder)

        jpg_files, data_folder = process_folder(folder)
        nr_jpg_files = len(jpg_files)
        jpg_files_id = np.arange(nr_jpg_files)

        # Break the list into smaller chunks of 100 images and process the chunks sequentially
        chunk_starts = np.arange(0, nr_jpg_files, CHUNK_SIZE)

        for n_chunk, ix_chunk in enumerate(chunk_starts):
            
            ix_chunk_start = ix_chunk
            ix_chunk_stop = np.min([ix_chunk+CHUNK_SIZE-1, nr_jpg_files-1])
            
            print('Chunk %d of %d, start %d, stop %d' % (n_chunk, len(chunk_starts),
                                                 ix_chunk_start, ix_chunk_stop))

            process_and_save_chunk(n_chunk, ix_chunk_start, ix_chunk_stop, jpg_files,
                     jpg_files_id, args.plate_format, nr_processes, data_folder, start)

        # Now merge all chunks and delete the temp files
        all_data = pd.DataFrame()

        for n_chunk, ix_chunk in enumerate(chunk_starts):

            path_to_this_chunk_file = os.path.join(
                data_folder, format('chunk%d_data.p' % n_chunk))
            chunk_data = load(path_to_this_chunk_file, verbose=False)
            all_data = all_data.append(chunk_data, ignore_index=True)

            # Remove the temp "chunk" file
            os.remove(path_to_this_chunk_file)

        print('Printing data to %s' % data_folder)

        # Print all data
        path_to_all_data_file = os.path.join(data_folder, 'all_data.txt')
        all_data.to_csv(path_to_all_data_file, sep='\t', index=False)

        # Print the file_id to path map
        jpg_map = pd.DataFrame(
            data={'path': jpg_files, 'file_id': jpg_files_id})
        path_to_jpg_map_file = os.path.join(data_folder, 'jpg_map.txt')
        jpg_map.to_csv(path_to_jpg_map_file, sep='\t', index=False)
