cimport numpy as np
import numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport log, sin


ctypedef fused int_or_long:
    np.int64_t
    np.int32_t

ctypedef fused float_or_double:
    np.float32_t
    np.float64_t



@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss(int_or_long [::1] y_true,
                          float_or_double[::1] exp_pred):

    cdef:
        np.float64_t eps = 1e-20
        np.float64_t log_loss_ = 0.
        np.float64_t loss = 0
        np.uint32_t i = 0
        np.uint32_t j = 0
        np.float64_t P_hat = 0.
        np.float64_t P = 0.
        np.float64_t cur_log_loss = 0.
        np.uint32_t size = y_true.shape[0]
    

    for i in prange(size, nogil=True, schedule='dynamic', num_threads=12):
        for j in range(i + 1):
            P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5
            P = 1.0 / (1.0 + (exp_pred[j] / exp_pred[i]))
            cur_log_loss = P_hat*log(P + eps) + (1.0 - P_hat)*log(1.0 - P + eps)
            if i == j:
                cur_log_loss *= 0.5
            loss += cur_log_loss / size**2
    return loss


@cython.boundscheck(False)
@cython.wraparound(False)
def test_squared(float_or_double [::1] y_pred):
    cdef:
        np.float64_t ssum = 0
        np.int64_t i = 0
        np.int64_t size = y_pred.shape[0]
        np.int64_t j = 0

    for i in prange(size**2, nogil=True, schedule='dynamic', num_threads=12):
        
        ssum += sin(y_pred[i%size] + y_pred[j//size])
    return ssum
