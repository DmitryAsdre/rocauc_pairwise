#import numpy as np
#from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
cimport cython

ctypedef fused int_or_long:
    np.int64_t
    np.int32_t

ctypedef fused float_or_double:
    np.float32_t
    np.float64_t



@cython.boundscheck(False)
@cython.wraparound(False)
def get_non_unique_labels_count(int_or_long [::1] y_true, 
                                float_or_double [::1] y_pred, 
                                np.int64_t [::1] y_pred_argsorted):
    cdef:
        np.int32_t counter_p = 0 
        np.int32_t counter_n = 0

        np.uint32_t l_pointer = 0
        np.uint32_t r_pointer = 0
    
        np.int32_t [::1] counters_p = np.zeros_like(y_true, dtype=np.int32)
        np.int32_t [::1] counters_n = np.zeros_like(y_true, dtype=np.int32)

    for r_pointer in range(len(y_pred)):
        if y_true[y_pred_argsorted[r_pointer]] == 1:
            counter_p += 1
        else:
            counter_n += 1
        if r_pointer + 1 !=  len(y_pred) and y_pred[y_pred_argsorted[r_pointer]] == y_pred[y_pred_argsorted[r_pointer + 1]]:
            continue
        else:
            while l_pointer <= r_pointer:
                counters_p[y_pred_argsorted[l_pointer]] = counter_p
                counters_n[y_pred_argsorted[l_pointer]] = counter_n
                l_pointer+=1
            counter_n = 0
            counter_p = 0
    return counters_p, counters_n



@cython.boundscheck(False)
@cython.wraparound(False)
def get_non_unique_borders(float_or_double [::1] y_pred,
                           np.int64_t [::1] y_pred_argsorted):

    cdef:
        np.int32_t [::1] y_pred_left = np.zeros_like(y_pred, dtype=np.int32)
        np.int32_t [::1] y_pred_right = np.zeros_like(y_pred, dtype=np.int32)
    
    cdef:
        np.int32_t left_p = 0
        np.int32_t j = 0
    
    for j in range(len(y_pred_argsorted)):
        if j + 1 != len(y_pred_argsorted) and y_pred[y_pred_argsorted[j]] == y_pred[y_pred_argsorted[j + 1]]:
            y_pred_left[y_pred_argsorted[j]] = left_p
        else:
            y_pred_left[y_pred_argsorted[j]] = left_p
            left_p = j + 1
    cdef:        
        np.int32_t right_p = len(y_pred_argsorted) - 1
    for j in range(len(y_pred_argsorted) - 1, -1, -1):
        if j - 1 != -1 and y_pred[y_pred_argsorted[j]] == y_pred[y_pred_argsorted[j - 1]]:
            y_pred_right[y_pred_argsorted[j]] = right_p
            continue
        else:
            y_pred_right[y_pred_argsorted[j]] = right_p
            right_p  = j - 1
            
    return y_pred_left, y_pred_right

@cython.boundscheck(False)
@cython.wraparound(False)
def get_inverse_argsort(int_or_long [::1] y_true, 
                        float_or_double [::1] y_pred):
    cdef:
        np.int64_t [::1] y_pred_argsorted = np.argsort(y_pred)[::-1]
        np.int64_t [::1] y_pred_ranks = np.zeros_like(y_true, dtype=np.int64_t)

        np.int64_t k = 0

    for k in range(len(y_pred_argsorted)):
        y_pred_ranks[y_pred_argsorted[k]] = k
    return y_pred_ranks

@cython.boundscheck(False)
@cython.wraparound(False)
def get_labelscount_borders(int_or_long [::1] y_true, 
                            float_or_double [::1] y_pred,
                            np.int64_t [::1] y_pred_argsorted):    
    cdef:
        np.int32_t[::1] counters_n
        np.int32_t[::1] counters_p

        np.int32_t[::1] y_pred_left
        np.int32_t[::1] y_pred_right

    counters_p, counters_n = get_non_unique_labels_count(y_true, y_pred, y_pred_argsorted)
    y_pred_left, y_pred_right = get_non_unique_borders(y_pred, y_pred_argsorted)
    
    return counters_p, counters_n, y_pred_left, y_pred_right