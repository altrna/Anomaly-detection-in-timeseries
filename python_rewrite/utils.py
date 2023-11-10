from datetime import datetime, timezone, timedelta
import numpy as np
import csv



def import_csv_data(csv_file):
    """
    loads up data from csv file
    args:
        csv_file : str - filename
    returns:
        keys : list of strings - name of columns
        val_matrix : np.array - numpy array of loaded data
    """
    f = open(csv_file)
    csv_val = list(csv.reader(f))
    keys = csv_val[0]
    val_matrix : np.ndarray = np.array(csv_val[1:]).astype(np.float64)
    return keys, val_matrix

def fix_time(valmatrix):
    """
    Changes unix time (default in data) to timeseries starting from 0
    args:
        valmatrix : np.array
    returns:
        valmatrix : np.array
    """
    valmatrix[:,1] = valmatrix[:,1] - valmatrix[0,1]*np.ones_like(valmatrix[:,1])
    return valmatrix

def choose_column(keys, param):
    return keys.index(param) if param in keys else print("WARN: given param is not in keys!")

def normalize_signal(signal, mean = None, sigma = None):
    """
    Normalizes each variable in signal (not time and sample no.)
    args:
        signal : np.array - signal to be normalized
        mean : np.array | None - mean of each variable in signal
        sigma : np.array | None - standard deviation of each variable in signal
        if mean or sigma is None, computes the values from the signal itself.
    returns:
        normalized_signal : np.array
    """
    if not isinstance(mean, np.ndarray):
        mean = np.mean(signal, axis = 0)
    if not isinstance(sigma, np.ndarray):
        sigma = np.std(signal, axis = 0)
    normalized_signal = np.hstack((signal[:, :2],(signal[:, 2:] - mean[2:])/sigma[2:]))
    return normalized_signal