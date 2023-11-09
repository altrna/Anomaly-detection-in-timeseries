import numpy as np
import matplotlib.pyplot as plt
from fastdtw import *
from utils import *
from time import sleep


DATA = ["./data/TireAssemblyFT disc.csv", "../data/TireAssemblyFT_1.csv", "../data/TireAssemblyFT_2.csv", "../data/TireAssemblyFT_3.csv", "../data/TireAssemblyFT_4.csv"]

def load_data_from_data_pair(data_pair : tuple):
    """Loads data from files specified by 'data_pair' tuple
    args:
        data_pair -> tuple of indices in DATA list above (global variable in this class)
    returns:
        valmatrix1, valmatrix2 -> two numpy arrays (timeseries) with all variables and time starting from 0"""
    keys, valmatrix1 = import_csv_data(DATA[data_pair[0]])
    keys, valmatrix2 = import_csv_data(DATA[data_pair[1]])
    valmatrix1 = fix_time(valmatrix1)
    valmatrix2 = fix_time(valmatrix2)
    return valmatrix1, valmatrix2
    

def compute_multivar_dtw(valmatrix1 : np.ndarray, valmatrix2 : np.ndarray, fast : bool, rad : int = 20):
    """Function that computes dtw for all dimensions of timeseries
        args:
            valmatrix1, valmatrix2 -> numpy array of two timeseries (time in the second column (index [:, 1])) other dimensions in >1 indices
            fast -> bool True if fastdtw is used
            rad -> int if fastdtw is used specifies the radius of searching- the higher the more precise, but slower
    """
    time1 = valmatrix1[:,1]
    time2 = valmatrix2[:,1]
    values = []
    for i in range(2, np.shape(valmatrix1)[1]):
        signal1 = valmatrix1[:,i]
        signal2 = valmatrix2[:,i]
        values.append(compute_dtw(time1, time2, signal1, signal2, fast = fast, rad = rad))
    return values

def compute_dtw(time1, time2, signal1, signal2, fast = True, rad = 5):
    """
    Computes euclidean distance of dtw of two timeseries
    inputs:
    time1 -> np.array
    time2 -> np.array
    signal1 -> np.array 
    signal2 -> np.array

    returns:
    distance: euclidean distance -> int
    paired_indices: np array of tuples of paired indices -> np.array
    """
    assert len(time1) == len(signal1)
    assert len(time2) == len(signal2)
    time_series1 = np.vstack((time1, signal1)).T
    time_series2 = np.vstack((time2, signal2)).T
    if fast:
        distance, paired = fastdtw(signal1, signal2, radius=rad)

    else:
        distance, paired = dtw(signal1, signal2)
    paired_indices = np.array(paired)
    return distance, paired_indices, time_series1, time_series2

def plot_dtw_vec(time_series1, time_series2, paired_indices, distance, newfig = False):
    """Implements vectorized visualization of dtw between two signals
        args:
            time_series1, time_series2 -> np.array (nx2) where the 0-th column is time and 1st is data
            paired_indices -> list of tuples of len 2 of paired indices using dtw
            distance -> int - euclidean distance of aligned timeseries
            newfig -> bool TODO for now it does nothing
            """
            
    if np.shape(time_series1)[0] > np.shape(time_series2)[0]:
        leading = time_series1
        l = 0
        s = 1
        short = time_series2
    else:
        leading = time_series2
        l = 1
        s = 0
        short = time_series1
    unique_indices_s = np.unique(paired_indices[:, s], return_index=True)[1]
    unique_indices_short = sorted(unique_indices_s)
    unique_indices_l = np.unique(paired_indices[:, l], return_index=True)[1]
    unique_indices_long = sorted(unique_indices_l)
    if newfig:
        plt.figure()
    plt.plot(leading[paired_indices[unique_indices_long, l], 0], leading[paired_indices[unique_indices_long, l], 1], label='Line Plot', linewidth=1, color="black")
    plt.plot(short[paired_indices[unique_indices_short, l], 0], short[paired_indices[unique_indices_short, s], 1], label='Line Plot', linewidth=1, color = "red")
    plt.title(f"Distance of signals: {distance}")
    return leading


def make_dtw_graph(time_series1, time_series2, dtw_paired_indices, distance, vectorized = True):
    """
    Creates matplotlib figure of two subplots the first shows both timeseries unaligned and the second aligned using DTW
    args:
        time_series1, time_series2 -> np.array (nx2) where the 0-th column is time and 1st is data
            paired_indices -> list of tuples of len 2 of paired indices using dtw
            distance -> int - euclidean distance of aligned timeseries
            vectorized -> bool implements vectorized computation if True (no reason not to do that now that it is implemented)
    """
    plt.figure()
    plt.subplot(211)
    len_1, len_2 = np.shape(time_series1)[0], np.shape(time_series2)[0]
    t1_c, t2_c = ("black", "red") if len_1 > len_2 else ("red", "black")
    plt.plot(time_series1[:, 0], time_series1[:, 1], label='Line Plot', linewidth=1, color = t1_c)
    plt.plot(time_series2[:, 0], time_series2[:, 1], label='Line Plot', linewidth=1, color = t2_c)
    plt.subplot(212)
    if not vectorized:
        for i, j in dtw_paired_indices:
            plt.plot([time_series1[i, 0], time_series2[j, 0]], [time_series1[i, 1], time_series2[j, 1]], color='red', linewidth=0.3, linestyle='--')
    else:
        plot_dtw_vec(time_series1, time_series2, dtw_paired_indices, distance)
    plt.show()


def plot_dtw(signal1, signal2, time1, time2, distances = False):
    """Legacy function probably not working..."""
    plt.figure()
    time_series1 = np.vstack((time1, signal1)).T
    time_series2 = np.vstack((time2, signal2)).T
    distance, paired = fastdtw(signal1, signal2)
    print(paired)
    paired_indices = np.array(paired)
    if np.shape(time_series1)[0] > np.shape(time_series2)[0]:
        leading = time_series1
        plt.plot(time1, signal1, label='Line Plot', linewidth=0.3)
        for i, j in paired_indices:
            if distances:
                plt.plot([leading[i, 0], leading[i, 0]], [time_series1[i, 1], time_series2[j, 1]], linewidth=0.3)
            
            if i > 0 and j > 0:
                plt.plot([leading[i - 1, 0], leading[i, 0]], [last_val, time_series2[j, 1]], linewidth=0.3, color="orange")
            last_val = time_series1[i, 1]
    else:
        leading = time_series2
        plt.plot(time2, signal2, label='Line Plot', linewidth=0.3)
        for i, j in paired_indices:
            if distances:
                plt.plot([leading[j, 0], leading[j, 0]], [time_series1[i, 1], time_series2[j, 1]], linewidth=0.3)

            if i > 0 and j > 0:
                plt.plot([leading[j - 1, 0], leading[j, 0]], [last_val, time_series1[i, 1]], linewidth=0.3, color="blue")
            last_val = time_series1[i, 1]



def compare_signals(valmatrix1, valmatrix2, start_idx = None, stop_idx = None, time_shift = None, new_figure = False, fast = True, radius = 5, vectorized = True):
    """
    main function that implements and plots the DTW of multidimensional system
    args:
        valmatrix1, valmatrix2 -> two np.arrays of data
        start_idx -> int | None - specifies the start of dtw algorithm - now as an index in valmatrix TODO for specific time
                        if None the computation is done from the start of timeseries
        stop_idx -> int | None - specifies the stop of dtw algorithm - now as an index in valmatrix TODO for specific time
                        if None the computation is done till the end of timeseries
        time_shift -> int | None - adds current time shift to values (if computing by parts) adds constant to time in timeseries
                        if None keeps time as is
        new_figure -> bool - if new figure should be opened TODO probably doesnt work
        fast -> bool True if fastdtw is used
        rad -> int if fastdtw is used specifies the radius of searching- the higher the more precise, but slower
    """
    if start_idx != None:
        valmatrix1 = valmatrix1[start_idx : , :]
        valmatrix2 = valmatrix2[start_idx : , :]
    if stop_idx != None:
        valmatrix1 = valmatrix1[ : stop_idx, :]
        valmatrix2 = valmatrix2[ : stop_idx, :]
    if time_shift != None:
        valmatrix1[:, 1] += time_shift
        valmatrix2[:, 1] += time_shift
    values = compute_multivar_dtw(valmatrix1, valmatrix2, fast=fast, rad=radius)
    ndim = 7
    if new_figure:
        plt.figure()
    else:
        plt.draw()
    n = 1
    for i in range(0, ndim):
        plt.subplot(ndim, 2, i + n)
        plt.plot(values[i][2][:, 0], values[i][2][:, 1], color = "red", linewidth = 1)
        plt.plot(values[i][3][:, 0], values[i][3][:, 1], color = "black", linewidth = 1)
        plt.subplot(ndim, 2, i + n + 1)
        n += 1
        plot_dtw_vec(values[i][2], values[i][3],values[i][1], values[i][0])
    plt.show()
    

