cimport numpy as np
np.import_array()
import numpy as np
cimport cython
from libcpp.utility cimport pair
from libc.stdint cimport int32_t


ctypedef fused int_t:
    np.int64_t
    np.int32_t

ctypedef fused _int_t:
    np.int64_t
    np.int32_t

ctypedef fused float_t:
    np.float32_t
    np.float64_t

cdef extern from '../src/cpu/deltaauc.cpp':
    double deltaauc[T_true, T_predranks](T_true* y_true, T_predranks* y_pred_ranks, 
                                        size_t n_ones, size_t n_zeroes, size_t i, size_t j)
    double deltaauc_exact[T_true, T_pred](T_true* y_true, T_pred* y_pred, 
                                         int32_t* counters_p, int32_t* counters_n,
                                         int32_t* y_pred_left, int32_t* y_pred_right,
                                         size_t n_ones, size_t n_zeroes,
                                         size_t i, size_t j)
    


@cython.boundscheck(False)
@cython.wraparound(False)
def deltaauc_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                np.ndarray[_int_t, ndim=1, mode='c'] y_pred_ranks,
                size_t n_ones, size_t n_zeroes, 
                size_t i, size_t j):
    cdef:
        double deltaauc_ = 0
    
    deltaauc_ = deltaauc(&y_true[0], &y_pred_ranks[0], n_ones, n_zeroes, i, j)

    return deltaauc_


@cython.boundscheck(False)
@cython.wraparound(False)
def deltaauc_exact_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                      np.ndarray[float_t, ndim=1, mode='c'] y_pred,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] counters_p,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] counters_n,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_left,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_right,
                      size_t n_ones, size_t n_zeroes, size_t i, size_t j):
    cdef:
        double deltauc_ = 0
    
    deltaauc_ = deltaauc_exact(&y_true[0], &y_pred[0], 
                               &counters_p[0], &counters_n[0],
                               &y_pred_left[0], &y_pred_right[0],
                               n_ones, n_zeroes, i, j)
    
    return deltaauc_

