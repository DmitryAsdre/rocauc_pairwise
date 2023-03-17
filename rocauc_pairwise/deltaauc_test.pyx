cimport numpy as np
import numpy as np

from deltaauc_cpu cimport deltaauc
cimport cython
from cython.parallel cimport prange


ctypedef fused int_t:
    np.int64_t
    np.int32_t

ctypedef fused float_t:
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_auc_exact_cpu(int_t [::1] y_true,
                                        float_t[::1] exp_pred,
                                        int_t num_threads):
    cdef:
        np.int64_t [::1] y_pred_argsorted =  np.argsort(exp_pred)        
    deltaauc_ij = deltaauc(y_true, y_pred_argsorted, 2, 2, 1, 2)
    return deltaauc_ij