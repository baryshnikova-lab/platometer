"""Code from https://github.com/demotu/BMC: to detect peaks in data based
on their amplitude and other features.
"""

from __future__ import division, print_function

import scipy
import numpy as np
import matplotlib.pyplot as plt


__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.4"
__license__ = "MIT"


def detect_peaks(data, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, axes=None):
    """Detects peaks in data based on their amplitude and other features.

    Args:
        data:   1D array_like
        mph: {None, number}, optional (default = None)
                detect peaks that are greater than minimum peak height.
        mpd: positive integer, optional (default = 1)
                detect peaks that are at least separated by minimum peak distance (in
                number of data).
        threshold: positive number, optional (default = 0)
                detect peaks (valleys) that are greater (smaller) than `threshold`
                in relation to their immediate neighbors.
        edge: {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
                for a flat peak, keep only the rising edge ('rising'), only the
                falling edge ('falling'), both edges ('both'), or don't detect a
                flat peak (None).
        kpsh: bool, optional (default = False)
                keep peaks with same height even if they are closer than `mpd`.
        valley: bool, optional (default = False)
                if True (1), detect valleys (local minima) instead of peaks.
        show: bool, optional (default = False)
                if True (1), plot data in matplotlib figure.
        axes: a matplotlib.axes.Axes instance, optional (default = None).

    Returns:
        ind : 1D array_like. Indices of the peaks in `data`.

    Notes:
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-data)`

        The function can handle NaN's.

        See this IPython Notebook [1]_.

    References:
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """

    data = np.atleast_1d(data).astype('float64')

    if data.size < 3:
        return np.array([], dtype=int)

    if valley:
        data = -data

    # Find indices of all peaks
    data_dx = data[1:] - data[:-1]

    # Handle NaN's
    indnan = np.where(np.isnan(data))[0]

    if indnan.size:
        data[indnan] = np.inf
        data_dx[np.where(np.isnan(data_dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)

    if not edge:
        ine = np.where((np.hstack((data_dx, 0)) < 0) & (np.hstack((0, data_dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((data_dx, 0)) <= 0) & (np.hstack((0, data_dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((data_dx, 0)) < 0) & (np.hstack((0, data_dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # Handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]

    # First and last values of data cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == data.size - 1:
        ind = ind[:-1]

    # Remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[data[ind] >= mph]

    # Remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        data_dx = np.min(np.vstack([data[ind] - data[ind - 1], data[ind] - data[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(data_dx < threshold)[0])

    # Detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(data[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # Keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (data[ind[i]] > data[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak

        # Remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            data[indnan] = np.nan
        if valley:
            data = -data
        _plot(data, mph, mpd, threshold, edge, valley, axes, ind)

    return ind


def _plot(data, mph, mpd, threshold, edge, valley, axes, ind):
    """Plots the results of the detect_peaks function.
    """

    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(8, 4))

    axes.plot(data, 'b', lw=1)

    if ind.size:
        label = 'valley' if valley else 'peak'
        label = label + 's' if ind.size > 1 else label
        axes.plot(ind, data[ind], '+', mfc=None, mec='r',
                  mew=2, ms=8, label='%d %s' % (ind.size, label))
        axes.legend(loc='best', framealpha=.5, numpoints=1)

    axes.set_xlim(-.02 * data.size, data.size * 1.02 - 1)
    ymin, ymax = data[np.isfinite(data)].min(), data[np.isfinite(data)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    axes.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    axes.set_xlabel('Data #', fontsize=14)
    axes.set_ylabel('Amplitude', fontsize=14)
    mode = 'Valley detection' if valley else 'Peak detection'
    axes.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                 % (mode, str(mph), mpd, str(threshold), edge))
    # plt.grid()
    plt.show()


def bucket(data, bucket_size):
    """Replaces groups of N consecutive pixels in
    the array with a single pixel which is the sum of the N replaced pixels.

    Args:
        data (numpy array): Image data.
        bucket_size (list or numpy array): Shape of 2D window that defines the bucket.
    See: http://stackoverflow.com/q/36269508/513688
    Author: Andrew York
    """

    for bucket_dim in bucket_size:
        assert float(bucket_dim).is_integer()

    bucket_size = [int(bucket_dim) for bucket_dim in bucket_size]
    data = np.ascontiguousarray(data)
    new_shape = np.concatenate((np.array(data.shape) // bucket_size, bucket_size))
    old_strides = np.array(data.strides)
    new_strides = np.concatenate((old_strides * bucket_size, old_strides))
    axis = tuple(range(data.ndim, 2 * data.ndim))
    return np.lib.stride_tricks.as_strided(data, new_shape, new_strides).sum(axis)


def fit_sin(x_vals, y_vals, guess=np.array([])):
    """
    Fits a sin function to the input sequence of x and y values.

    Args:
        x_vals (numpy array): x-values of the input sequence.
        y_vals (numpy array): y-values of the input sequence.
        guess (numpy array, optional): An initial guess of the sinusoid
            parameters (helps with robustness).

    Returns:
        dict: A dictionary containing the fitting parameters.
    """

    amplitude_guess, angular_frequency_guess = None, None
    phase_guess, offset_guess, slope_guess = None, None, None

    if guess.any():
        amplitude_guess, angular_frequency_guess, phase_guess, offset_guess, slope_guess = guess

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    if not amplitude_guess:
        amplitude_guess = np.std(y_vals) * 2. ** 0.5

    if not offset_guess:
        offset_guess = np.mean(y_vals)

    if not slope_guess:
        slope_guess = 0.01

    if not phase_guess:
        phase_guess = 0.0

    if not angular_frequency_guess:
        # Assume uniform spacing
        fourier_frequencies = np.fft.fftfreq(len(x_vals), (x_vals[1]-x_vals[0]))
        fourier_transforms = abs(np.fft.fft(y_vals))

        # Excluding the zero frequency "peak", which is related to offset
        frequency_guess = abs(fourier_frequencies[np.argmax(fourier_transforms[1:])+1])
        angular_frequency_guess = 2. * np.pi * frequency_guess

    guess = np.array([amplitude_guess, angular_frequency_guess,
                      phase_guess, offset_guess, slope_guess])

    def sinfunc(time, amplitude, angular_frequency, phase, offset, slope):
        return amplitude * np.sin(angular_frequency*time + phase) + offset + slope*time

    [popt, pcov] = scipy.optimize.curve_fit(sinfunc, x_vals, y_vals, p0=guess)
    amplitude, angular_frequency, phase, offset, slope = popt
    frequency = angular_frequency/(2.*np.pi)
    fitfunc = lambda t: amplitude * np.sin(angular_frequency*t + phase) + offset + slope*t

    return {"a": amplitude,
            "w": angular_frequency,
            "p": phase,
            "c": offset,
            "d": slope,
            "freq": frequency,
            "popt": popt,
            "period": 1./frequency,
            "fitfunc": fitfunc,
            "maxcov": np.max(pcov),
            "rawres": (guess, popt, pcov)}
