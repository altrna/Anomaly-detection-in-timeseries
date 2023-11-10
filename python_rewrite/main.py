import numpy as np
import matplotlib.pyplot as plt
from fastdtw import *
from utils import *
from time import sleep
from scipy.signal import correlate

DATA = ["./data/TireAssemblyFT disc.csv", "../data/TireAssemblyFT_1.csv", "../data/TireAssemblyFT_2.csv", "../data/TireAssemblyFT_3.csv", "../data/TireAssemblyFT_4.csv"]

def load_data_from_tuple(data_to_load : tuple):
    """Loads data from files specified by 'data_pair' tuple
    args:
        data_pair : tuple of indices in DATA list above (global variable in this class)
    returns:
        valmatrix1, valmatrix2 : two numpy arrays (timeseries) with all variables and time starting from 0"""
    matrices = list()
    for i in range(len(data_to_load)):
        keys, valmatrix1 = import_csv_data(DATA[data_to_load[i]])
        valmatrix1 = fix_time(valmatrix1)
        matrices.append(valmatrix1)
    return matrices
    

def compute_multivar_dtw(valmatrix1 : np.ndarray, valmatrix2 : np.ndarray):
    """Function that computes dtw for all dimensions of timeseries
        args:
            valmatrix1, valmatrix2 : numpy array of two timeseries (time in the second column (index [:, 1])) other dimensions in >1 indices
    """
    dist, paired_indices = dtw(valmatrix1[:, 2:], valmatrix2[:, 2:])
    paired  = np.array(paired_indices)
    return dist, paired

def compute_dtw(time1, time2, signal1, signal2, fast = False, rad = 5):
    """
    Computes euclidean distance of dtw of two timeseries
    inputs:
    time1 : np.array
    time2 : np.array
    signal1 : np.array 
    signal2 : np.array

    returns:
    distance: euclidean distance : int
    paired_indices: np array of tuples of paired indices : np.array
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

def plot_dtw(signal1, signal2, time1, time2, distances = False):
    """Legacy function probably not working..."""
    plt.figure()
    time_series1 = np.vstack((time1, signal1)).T
    time_series2 = np.vstack((time2, signal2)).T
    distance, paired = dtw(signal1, signal2)
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

def plot_dtw_vec(time_series1, time_series2, paired_indices, distance, axes = plt.Axes, time = 1, variable = 0):
    """Implements vectorized visualization of dtw between two signals
        args:
            time_series1, time_series2 : np.array (nx2) where the 0-th column is time and 1st is data
            paired_indices : list of tuples of len 2 of paired indices using dtw
            distance : int - euclidean distance of aligned timeseries
            newfig : bool TODO for now it does nothing
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
    axes.plot(leading[paired_indices[unique_indices_long, l], time], leading[paired_indices[unique_indices_long, l], variable], label='Line Plot', linewidth=1, color="black")
    axes.plot(short[paired_indices[unique_indices_short, l], time], short[paired_indices[unique_indices_short, s], variable], label='Line Plot', linewidth=1, color = "red")
    return leading

def compare_signals(valmatrix1, valmatrix2, start_idx = None, stop_idx = None, time_shift = None, new_figure = False, fast = True, radius = 5, vectorized = True):
    """
    main function that implements and plots the DTW of multidimensional system
    args:
        valmatrix1, valmatrix2 : two np.arrays of data
        start_idx : int | None - specifies the start of dtw algorithm - now as an index in valmatrix TODO for specific time
                        if None the computation is done from the start of timeseries
        stop_idx : int | None - specifies the stop of dtw algorithm - now as an index in valmatrix TODO for specific time
                        if None the computation is done till the end of timeseries
        time_shift : int | None - adds current time shift to values (if computing by parts) adds constant to time in timeseries
                        if None keeps time as is
        new_figure : bool - if new figure should be opened TODO probably doesnt work
        fast : bool True if fastdtw is used
        rad : int if fastdtw is used specifies the radius of searching- the higher the more precise, but slower
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
    distance, paired = compute_multivar_dtw(valmatrix1, valmatrix2)
    ndim = 7
    n = 0
    fig = plt.figure()
    axes = [fig.add_subplot(7,2,i) for i in range(1, 2*ndim + 1)]
    for i in range(0, ndim):
        axes[i + n].plot(valmatrix1[:, 1], valmatrix1[ : , i + 2], color = "red", linewidth = 1)
        axes[i + n].plot(valmatrix2[:, 1], valmatrix2[ : , i + 2], color = "black", linewidth = 1)       
        plot_dtw_vec(valmatrix1, valmatrix2, paired, distance, axes=axes[i + n + 1], variable=i+2)
        n += 1
    plt.show()

def find_closest_part(signal_part, signal_whole, last_i = 0, step_size = 50):
    """Finds best-suiting part of reference signal to a part of signal that is tested
        args:
            signal_part : np.ndarray - part of signal to be tested
            signal_whole : np.ndarray - reference signal
            last_i : int - index of the max index of reference signal found in previous iteration
            step_size : int - step size to iterate through the refference signal
        returns:
            dist : int - distance of dtw aligned signal_part and properly chosen part of the refference signal
            new_last_i : int - max index in the refference signal for future iterations
    """

    n = np.shape(signal_whole)[0]
    dist = []
    length = []
    for i in range(max(n//20, last_i + step_size//2), n, max(n//20, 1)):
        cur_dist, _ = dtw(signal_part[:, 2:], signal_whole[:i, 2:])
        dist.append(cur_dist)
        length.append(i)
    min_idx = np.argmin(dist)
    for i in range(max(min_idx - 1, 1), min(min_idx + 1, len(length))):
        cur_dist, _ = dtw(signal_part[:, 2:], signal_whole[:length[i], 2:])
        dist.append(cur_dist)
        length.append(i)
    min_idx = np.argmin(dist)
    new_last_i = length[min_idx]
    return dist, new_last_i
    
def cross_validate_signals(signal1, signals, step_size = 50):
    """
    1. randomly selects a signal from dataset, normalizes it and by certain step size compares it to the tested signal
    
    2. This function should simulate comparing continuous flow of data (signal1) by certain step size and comparing it to
    the normalized reference signal. It compares each iteration and finds optimal dtw alignment. This is shown in matplotlib
    window which contains a ndim * 2 subplot (in our case ndim=7)

    also it saves the distance metric which is at this time not used, but will be when the real "cross validation" will be implemented

    args:
        signal1 : np.array - signal to be tested
        signals : list of np.arrays - database of signals to test on
        step_size : length of iterations
    returns:
        None
    """
    all_signals = np.vstack(signals)
    mean_ = np.mean(all_signals, axis = 0)
    sigma_ = np.std(all_signals, axis = 0)
    signal1 = normalize_signal(signal1)
    for s in range(len(signals)):
        signals[s] = normalize_signal(signals[s], mean = mean_, sigma = sigma_)
    chosen_int = np.random.randint(0, len(signals))
   
    chosen = signals[chosen_int]
    distances = []
    I = [0]
    ndim = 7
    length = np.shape(signal1)[0]
    fig = plt.figure()
    axes = [fig.add_subplot(7,2,i) for i in range(1, 2*ndim + 1)]
    for i in range(step_size, length, step_size):        
        signal1_part = signal1[:i, :]
        iopt = find_closest_part(signal1_part, chosen, last_i=I[-1], step_size=step_size)[1]
        signal2_part = chosen[:iopt, :]
        dist, paired = dtw(signal1_part, signal2_part)
        paired = np.array(paired)
        distances.append(dist)
        I.append(iopt)
        n = 0
        for j in range(0, ndim):
            axes[j + n].clear()
            axes[j + n].plot(signal1_part[:, 1], signal1_part[ : , j + 2], color = "red", linewidth = 1)
            axes[j + n].plot(signal2_part[:, 1], signal2_part[ : , j + 2], color = "black", linewidth = 1)
            axes[j + n + 1].clear()
            plot_dtw_vec(signal1_part, chosen, paired, dist, axes[j + n + 1], variable=j+2)
            n += 1
        plt.pause(0.05)
        plt.show()

