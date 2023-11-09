from datetime import datetime, timezone, timedelta
import numpy as np
import csv



def import_csv_data(csv_file):
    f = open(csv_file)
    csv_val = list(csv.reader(f))
    keys = csv_val[0]
    val_matrix : np.ndarray = np.array(csv_val[1:]).astype(np.float64)
    return keys, val_matrix

def fix_time(valmatrix):
    valmatrix[:,1] = valmatrix[:,1] - valmatrix[0,1]*np.ones_like(valmatrix[:,1])
    return valmatrix

def choose_column(keys, param):
    return keys.index(param) if param in keys else print("WARN: given param is not in keys!")