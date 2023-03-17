cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport abs



@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t deltaauc_exact(int_t [::1] y_true,
                                 float_t [::1] y_pred,
                                 np.int32_t [::1] counters_p,
                                 np.int32_t [::1] counters_n,
                                 np.int32_t [::1] y_pred_left,
                                 np.int32_t [::1] y_pred_right,
                                 np.int32_t n_ones,
                                 np.int32_t n_zeroes,
                                 int_t i, 
                                 int_t j) nogil:
    cdef:
        float_t ypredi = y_pred[i]
        float_t ypredj = y_pred[j]
    
    if ypredi < ypredj:
        i, j = j, i

    ypredi = y_pred[i]
    ypredj = y_pred[j]

    cdef: 
        int_t li = y_true[i]
        int_t lj = y_true[j]


        np.float64_t deltaji = lj - li
    
    cdef:    
        np.float64_t deltai = 0.5 * counters_p[i]*counters_n[i] - 0.5*(counters_p[i] + deltaji) * (1.0 * counters_n[i] - deltaji)
        np.float64_t deltaj = 0.5 * counters_p[j]*counters_n[j] - 0.5*(counters_p[j] - deltaji) * (1.0 * counters_n[j] + deltaji)
    cdef:
        np.float64_t delta_eq = 0.
        np.float64_t multiplicate = 1.

    if deltaji == -1:
        delta_eq = counters_p[i] + counters_n[j] - 2
    else:
        delta_eq = -(counters_p[i] + counters_n[j])
    
    if deltaji == 0:
        multiplicate *= 0
    if ypredi == ypredj:
        multiplicate *= 0

    return multiplicate * (delta_eq + deltai + deltaj - deltaji * abs(y_pred_right[i] - y_pred_left[j])) / (n_ones * n_zeroes)

@cython.boundscheck(False)
@cython.wraparound(False)
def deltaauc_exact_wrapper (int_t [::1] y_true,
                            float_t [::1] y_pred,
                            np.int32_t [::1] counters_p,
                            np.int32_t [::1] counters_n,
                            np.int32_t [::1] y_pred_left,
                            np.int32_t [::1] y_pred_right,
                            np.int32_t n_ones,
                            np.int32_t n_zeroes,
                            int_t i, 
                            int_t j):

    return deltaauc_exact(y_true, y_pred, 
                          counters_p, counters_n, 
                          y_pred_left, y_pred_right, 
                          n_ones, n_zeroes, i, j)
           
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t deltaauc(int_t [::1] y_true,
                        np.int64_t [::1] y_pred_ranks,
                        np.int32_t n_ones,
                        np.int32_t n_zeroes,
                        np.int32_t i, np.int32_t j) nogil:
    cdef:
        np.float64_t i_ = 0
        np.float64_t j_ = 0
        np.float64_t deltaauc_ = 0.

    i_, j_ = y_pred_ranks[i], y_pred_ranks[j]

    deltaauc_ =  ((y_true[i] - y_true[j]) * (j_ - i_)) / (n_ones * n_zeroes)
    return deltaauc_

@cython.boundscheck(False)
@cython.wraparound(False)
def deltaauc_wrapper(int_t [::1] y_true,
                     np.int64_t [::1] y_pred_ranks,
                     np.int32_t n_ones,
                     np.int32_t n_zeroes,
                     np.int32_t i, np.int32_t j):
    return deltaauc(y_true, y_pred_ranks, n_ones, n_zeroes, i, j) 

