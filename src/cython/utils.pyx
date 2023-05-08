cimport numpy as np
np.import_array()
import numpy as np
cimport cython
from libcpp.utility cimport pair

cdef extern from '../src/cpu/tuple.hpp':
    cdef cppclass tuple[T_1, T_2, T_3, T_4]:
        pass

cdef extern from '../src/cpu/utils.hpp':
    pair[int*, int*] get_non_unique_labels_count[T](int* y_true, T* y_pred, long* y_pred_argsorted, size_t N)
    pair[int*, int*] get_non_unique_borders[T](T* y_pred, long* y_pred_argsorted, size_t N)
    tuple[int*, int*, int*, int*] get_labelscount_borders[T](int* y_true, T* y_pred, long* y_pred_argsorted, size_t N)
    long* get_inverse_argsort[T](int* y_true, T* y_pred, long* y_pred_argsorted, size_t N)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_non_unique_borders_py(np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred,
                              np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted):
    cdef:
        float* _y_pred = <float*>y_pred.data
        long* _y_pred_argsorted = <long*>y_pred_argsorted.data
        pair[int*, int*] non_unique_borders
    
    non_unique_borders = get_non_unique_borders(_y_pred, _y_pred_argsorted, y_pred.shape[0])

    cdef:
        np.npy_intp dims = y_pred.shape[0]
        np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_left = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT32, <void*>non_unique_borders.first)
        np.ndarray[np.int32_t, ndim=1, mode='c'] y_pred_right = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_INT32, <void*>non_unique_borders.second)
    
    return y_pred_left, y_pred_right

