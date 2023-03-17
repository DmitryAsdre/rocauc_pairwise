cimport numpy as cnp
import numpy as np
cimport cython
from cython.parallel cimport prange
from cython cimport parallel
from libc.math cimport log
cimport openmp
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libc.math cimport floor, sqrt, sin
from cython.view cimport array as cvarray

from rocauc_pairwise.deltaauc_cpu cimport deltaauc

cnp.import_array()

ctypedef fused int_t:
    cnp.int64_t
    cnp.int32_t

ctypedef fused float_t:
    cnp.float32_t
    cnp.float64_t


#cdef cnp.float64_t mult(cnp.int32_t [::1] i, cnp.int32_t [::1] j) nogil:
#    cdef:
#        cnp.float64_t [:, :, :] ssum =  cvarray(shape=i.shape, itemsize=sizeof(cnp.float64_t), format="i")

#        cnp.int32_t s = 0
#    for s in range(i.shape[0]):
#        ssum[s] += sin(i[s] + j[s])
#    return ssum

@cython.boundscheck(False)
@cython.wraparound(False)
def multp(int_t [::1] y_true, float_t [::1] y_pred):
    cdef:
        cnp.int64_t [::1] y_pred_argsorted =  np.argsort(exp_pred)
    return deltaauc(y_true, y_pred_argsorted, 10, 12, 0, 1)


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef deltaauc_test(int_t [::1] y_true,
#             np.int64_t [::1] y_pred_ranks,
#             np.int32_t n_ones,
#             np.int32_t n_zeroes,
#             np.int32_t i, np.int32_t j):
#    cdef:
#        np.float64_t i_ = 0
#        np.float64_t j_ = 0
#        float deltaauc_ = 0.

#    i_, j_ = y_pred_ranks[i], y_pred_ranks[j]

#    deltaauc_ =  ((y_true[i] - y_true[j]) * (j_ - i_)) / (n_ones * n_zeroes)
#    return deltaauc_

