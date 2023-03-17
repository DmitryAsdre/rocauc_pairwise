cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport abs


ctypedef fused int_t:
    np.int64_t
    np.int32_t

ctypedef fused float_t:
    np.float32_t
    np.float64_t


cdef np.float64_t deltaauc_exact(int_t [::1] y_true,
                                 float_t [::1] y_pred,
                                 np.int32_t [::1] counters_p,
                                 np.int32_t [::1] counters_n,
                                 np.int32_t [::1] y_pred_left,
                                 np.int32_t [::1] y_pred_right,
                                 np.int32_t n_ones,
                                 np.int32_t n_zeroes,
                                 int_t i, 
                                 int_t j) nogil

cdef np.float64_t deltaauc(int_t [::1] y_true,
                        np.int64_t [::1] y_pred_ranks,
                        np.int32_t n_ones,
                        np.int32_t n_zeroes,
                        np.int32_t i, np.int32_t j) nogil

