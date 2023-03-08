import numpy as np

from numba import jit


@jit(nopython=True)
def get_non_unique_labels_count(y_true, y_pred, y_pred_argsorted):
    counter_p, counter_n = 0, 0
    counters_p = np.zeros_like(y_true, dtype=np.int32)
    counters_n = np.zeros_like(y_true, dtype=np.int32)
    
    l_pointer = 0
    for r_pointer in range(len(y_pred)):
        if y_true[y_pred_argsorted[r_pointer]] == 1:
            counter_p += 1
        else:
            counter_n += 1
        if r_pointer + 1 != len(y_pred) and y_pred[y_pred_argsorted[r_pointer]] == y_pred[y_pred_argsorted[r_pointer + 1]]:
            continue
        else:
            while l_pointer <= r_pointer:
                counters_p[y_pred_argsorted[l_pointer]] = counter_p
                counters_n[y_pred_argsorted[l_pointer]] = counter_n
                l_pointer+=1
            counter_n = 0
            counter_p = 0
    return counters_p, counters_n

@jit(nopython=True)
def get_non_unique_borders(y_pred, y_pred_argsorted):
    y_pred_left = np.zeros_like(y_pred, dtype=np.int32)
    y_pred_right = np.zeros_like(y_pred, dtype=np.int32)
    
    left_p = 0
    for j in range(len(y_pred_argsorted)):
        if j + 1 != len(y_pred_argsorted) and y_pred[y_pred_argsorted[j]] == y_pred[y_pred_argsorted[j + 1]]:
            y_pred_left[y_pred_argsorted[j]] = left_p
        else:
            y_pred_left[y_pred_argsorted[j]] = left_p
            left_p = j + 1
            
    right_p = len(y_pred_argsorted) - 1
    for j in range(len(y_pred_argsorted) - 1, -1, -1):
        if j - 1 != -1 and y_pred[y_pred_argsorted[j]] == y_pred[y_pred_argsorted[j - 1]]:
            y_pred_right[y_pred_argsorted[j]] = right_p
            continue
        else:
            y_pred_right[y_pred_argsorted[j]] = right_p
            right_p  = j - 1
            
    return y_pred_left, y_pred_right

@jit(nopython=True)
def get_labelscount_borders(y_true, y_pred, y_pred_argsorted):    
    counters_p, counters_n = get_non_unique_labels_count(y_true, y_pred, y_pred_argsorted)
    y_pred_left, y_pred_right = get_non_unique_borders(y_pred, y_pred_argsorted)
    
    return counters_p, counters_n, y_pred_left, y_pred_right

@jit(nopython=True)
def get_inverse_argsort(y_true, y_pred):
    y_pred_argsorted = np.argsort(y_pred)[::-1]
    y_pred_ranks = np.zeros_like(y_true, dtype=np.int32)
    
    for k in range(len(y_pred_argsorted)):
        y_pred_ranks[y_pred_argsorted[k]] = k
    return y_pred_ranks