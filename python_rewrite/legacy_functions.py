import matplotlib.pyplot as plt
from main import *
from utils import *

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



def find_optimal_corelation(signal1, signal2):
    cross_cor = correlate(signal1, signal2, mode="full")
    optimal_offset = np.argmax(cross_cor) - (len(signal2) - 1)
    print(optimal_offset)
    return optimal_offset

def plot_correlation(signal1, signal2, time1, time2, offset):
    plt.figure()
    time_idx = range(max(len(signal1), len(signal2)) + abs(offset) + 1)
    if offset > 0:
        plt.plot(time_idx[:len(signal1)], signal1)
        plt.plot((time_idx + offset)[:len(signal2)], signal2)
    else:
        plt.plot((time_idx + offset)[:len(signal1)], signal1)
        plt.plot(time_idx[:len(signal2)], signal2)
    plt.show()