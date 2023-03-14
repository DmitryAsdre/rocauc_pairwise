cimport numpy as np
import numpy as np
cimport cython
from cython.parallel cimport prange
from cython cimport parallel
from libc.math cimport log
cimport openmp
from libc.stdlib cimport malloc, free
from libcpp cimport bool





ctypedef fused int_t:
    np.int64_t
    np.int32_t

ctypedef fused float_t:
    np.float32_t
    np.float64_t



@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss(int_t [::1] y_true,
                          float_t[::1] exp_pred,
                          int_t num_threads):

    cdef:
        np.float64_t eps = 1e-20
        np.float64_t loss = 0
        np.uint32_t i = 0
        np.uint32_t j = 0
        np.float64_t P_hat = 0.
        np.float64_t P = 0.
        np.float64_t cur_log_loss = 0.
        np.uint32_t size = y_true.shape[0]

    

    for i in range(size):#prange(size, nogil=True, schedule='dynamic', num_threads=num_threads):
        for j in range(i + 1):
            #P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5
            if y_true[i] == y_true[j]:
                P_hat = 0.5
            else:
                P_hat = float(y_true[i] > y_true[j])
            P = 1.0 / (1.0 + (exp_pred[j] / exp_pred[i]))
            cur_log_loss = P_hat*log(P + eps) + (1.0 - P_hat)*log(1.0 - P - eps)
            #if i == j:
            #    cur_log_loss *= 0.5
            loss += cur_log_loss
    return loss



@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess(int_t [::1] y_true,
                               float_t [::1] exp_pred,
                               int_t num_threads):
    cdef:
        np.float64_t cur_hess_i = 0.
        np.float64_t cur_grad_i = 0.
        np.uint32_t j = 0
        np.uint32_t i = 0
        np.uint32_t size = y_true.shape[0]
        openmp.omp_lock_t * locks = <openmp.omp_lock_t *> malloc(size * sizeof(openmp.omp_lock_t))

        np.float64_t exp_tmp_diff = 0.
        np.float64_t cur_d_dx_i = 0.
        np.float64_t cur_d_dx_j = 0.
        np.float64_t cur_d2_dx2_i = 0.
        np.float64_t cur_d2_dx2_j = 0.

        np.float64_t P_hat = 0.

        np.uint32_t P_GROSS = 13
        np.uint32_t P_SMALL = 173

        np.float64_t [::1] grad = np.zeros([size,], dtype=np.float64)
        np.float64_t [::1] hess = np.zeros([size,], dtype=np.float64)
 
    for l in range(size):
        openmp.omp_init_lock(&(locks[l]))

    for i in prange(size, nogil=True, schedule='dynamic', num_threads=num_threads):
        i = (P_SMALL * i) % size
        for j in range(i + 1, -1, -1):
            j = (P_GROSS * j) % (i + 1)
            exp_tmp_diff = exp_pred[i] / exp_pred[j]
            P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5
            cur_d_dx_i = ((P_hat - 1.) * exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.)
            cur_d_dx_j = - cur_d_dx_i
            cur_d2_dx2_i = -exp_pred[i]*exp_pred[j]*((exp_pred[j])/ (exp_tmp_diff + 1.)) ** 2
            cur_d2_dx2_j = cur_d2_dx2_i

            openmp.omp_set_lock(&(locks[j]))
            grad[j] += cur_d_dx_j
            hess[j] += cur_d2_dx2_j 
            openmp.omp_unset_lock(&(locks[j]))
        
            openmp.omp_set_lock(&(locks[i]))
            grad[i] += cur_d2_dx2_i
            hess[i] += cur_d2_dx2_j
            openmp.omp_unset_lock(&(locks[i]))
    free(locks)

    return np.asarray(grad), np.asarray(hess)