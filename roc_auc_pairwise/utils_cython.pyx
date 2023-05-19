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


cdef extern from '../src/cpu/utils.cpp':
    pair[int*, int*] get_non_unique_labels_count[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)
    pair[int*, int*] get_non_unique_borders[T_pred, T_argsorted](T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)
    long* get_inverse_argsort_wrapper[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)



@cython.boundscheck(False)
@cython.wraparound(False)
def get_inverse_argsort_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                           np.ndarray[float_t, ndim=1, mode='c'] y_pred,
                           np.ndarray[_int_t, ndim=1, mode='c'] y_pred_argsorted):

    cdef:
        long* _inversed_argsort

    _inversed_argsort = get_inverse_argsort_wrapper(&y_true[0], &y_pred[0], &y_pred_argsorted[0], y_true.shape[0])

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.int64_t, ndim=1, mode='c'] inversed_argsort = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT64, <void*>_inversed_argsort)

    return inversed_argsort


@cython.boundscheck(False)
@cython.wraparound(False)
def get_non_unique_borders_py(np.ndarray[float_t, ndim=1, mode='c'] y_pred,
                              np.ndarray[int_t, ndim=1, mode='c'] y_pred_argsorted):
    cdef:
        pair[int*, int*] non_unique_borders
    
    non_unique_borders = get_non_unique_borders(&y_pred[0], &y_pred_argsorted[0], y_pred.shape[0])

    cdef:
        np.npy_intp dims = y_pred.shape[0]
        np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_left = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT32, <void*>non_unique_borders.first)
        np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_right = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT32, <void*>non_unique_borders.second)
    
    return y_pred_left, y_pred_right

@cython.boundscheck(False)
@cython.wraparound(False)
def get_non_unique_labels_count_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                                   np.ndarray[float_t, ndim=1, mode='c'] y_pred,
                                   np.ndarray[_int_t, ndim=1, mode='c'] y_pred_argsorted):
    cdef:
        pair[int*, int*] non_unique_labels_count

    non_unique_labels_count = get_non_unique_labels_count(&y_true[0], &y_pred[0], &y_pred_argsorted[0], y_true.shape[0])

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.int32_t, ndim=1, mode='c'] counters_p = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT32, <void*>non_unique_labels_count.first)
        np.ndarray[np.int32_t, ndim=1, mode='c'] counters_n = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT32, <void*>non_unique_labels_count.second)

    return counters_p, counters_n