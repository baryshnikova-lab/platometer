import os
import dill as pickle
import pandas as pd


def save_to_p(data, output_file=None):

    if not output_file:
        output_file = os.path.join(os.getcwd(), 'output.p')

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle)


def load(file_path, v=3, verbose=True):

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
            ks = store.keys()

        for k in ks:
            k = k.lstrip('\/')
            if verbose:
                print(k)
            output[k] = pd.read_hdf(file_path, k)

        return output

    elif file_extension == '.p':

        if v == 3:
            output = pd.read_pickle(file_path)
        else:
            f = open(file_path, 'r')
            output = pickle.load(f)

        if verbose:
            print(', '.join([k for k in output.keys()]))
        return output

    else:
        print("Extension unknown. Only pickle (.p) and HDF5 (.h5) files are supported.")
