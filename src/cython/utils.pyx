cimport numpy as np
import numpy as np
cimport cython

cdef extern from '../src/cpu/utils.hpp':
    tuple[int*, int*] get_non_unique_labels_count[T](int* y_true, T* y_pred, long* y_pred_argsorted, size_t N)
    tuple[int*, int*] get_non_unique_borders[T](T* y_pred, long* y_pred_argsorted, size_t N)
    tuple[int*, int*, int*, int*] get_labelscount_borders[T](int* y_true, T* y_pred, long* y_pred_argsorted, size_t N)
    long* get_inverse_argsort[T](int* y_true, T* y_pred, long* y_pred_argsorted, size_t N)



