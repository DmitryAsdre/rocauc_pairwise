cimport numpy as np
np.import_array()
import numpy as np
cimport cython
from libc.stdint cimport int32_t

cdef extern from '../src/cuda/deltaauc.cuh':
    float deltaauc(int32_t* y_true, int32_t* y_pred_ranks,
                   size_t n_ones, size_t n_zeroes,
                   size_t i, size_t j, size_t N)
    float deltaauc_exact(int32_t* y_true, float* y_pred,
                         int32_t* counters_p, int32_t* counters_n,
                         int32_t* y_pred_left, int32_t* y_pred_right,
                         size_t n_ones, size_t n_zeroes, size_t i, size_t j, size_t N)

@cython.boundscheck(False)
@cython.wraparound(False)
def deltaauc_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_ranks,
                size_t n_ones, size_t n_zeroes,
                size_t i, size_t j):
    cdef:
        size_t N = y_true.shape[0]
    
        float _deltaauc = 0
    
    _deltaauc = deltaauc(&y_true[0], &y_pred_ranks[0], n_ones, n_zeroes, i, j, N)

    return _deltaauc

@cython.boundscheck(False)
@cython.wraparound(False)
def deltaauc_exact_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                      np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] counters_p,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] counters_n,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_left, 
                      np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_right,
                      size_t n_ones, size_t n_zeroes,
                      size_t i, size_t j):
    cdef:
        size_t N = y_true.shape[0]

        float _deltaauc = 0
    
    _deltaauc = deltaauc_exact(&y_true[0], &y_pred[0], &counters_p[0], &counters_n[0], 
                               &y_pred_left[0], &y_pred_right[0], n_ones, n_zeroes, i, j, N)

    return _deltaauc


