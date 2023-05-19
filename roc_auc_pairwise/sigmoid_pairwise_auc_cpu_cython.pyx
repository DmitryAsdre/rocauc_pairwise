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
    long* get_inverse_argsort[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)

cdef extern from '../src/cpu/deltaauc.cpp':
    double deltaauc[T_true, T_predranks](T_true* y_true, T_predranks* y_pred_ranks, 
                                        size_t n_ones, size_t n_zeroes, size_t i, size_t j)
    double deltaauc_exact[T_true, T_pred](T_true* y_true, T_pred* y_pred, 
                                         int32_t* counters_p, int32_t* counters_n,
                                         int32_t* y_pred_left, int32_t* y_pred_right,
                                         size_t n_ones, size_t n_zeroes,
                                         size_t i, size_t j)

cdef extern from '../src/cpu/sigmoid_pairwise_auc.cpp':
    double sigmoid_pairwise_loss_auc_cpu[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* exp_pred, 
                                                                      T_argsorted* y_pred_argsorted, 
                                                                      size_t n_ones, size_t n_zeroes, size_t N)
    double sigmoid_pairwise_loss_auc_exact_cpu[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* exp_pred, 
                                                                            T_argsorted* y_pred_argsorted, double eps,
                                                                            size_t n_ones, size_t n_zeroes, size_t N)
    pair[double*, double*] sigmoid_pairwise_diff_hess_auc_cpu[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* exp_pred,
                                                                                           T_argsorted* y_pred_argsorted,
                                                                                           size_t n_ones, size_t n_zeroes, size_t N)
    pair[double*, double*] sigmoid_pairwise_diff_hess_auc_exact_cpu[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* exp_pred,
                                                                                                 T_argsorted* y_pred_argsorted, double eps,
                                                                                                 size_t n_ones, size_t n_zeroes, size_t N)



@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_auc_cpu_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                                     np.ndarray[float_t, ndim=1, mode='c'] y_pred):
    cdef:
        double loss = 0
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()
        np.ndarray[float_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        size_t N = y_true.shape[0]
        size_t n_ones = np.sum(y_true)
        size_t n_zeroes = N - n_ones

    loss = sigmoid_pairwise_loss_auc_cpu(&y_true[0], &exp_pred[0], &y_pred_argsorted[0], n_ones, n_zeroes, N)

    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_auc_exact_cpu_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                                           np.ndarray[float_t, ndim=1, mode='c'] y_pred,
                                           double eps = 0):
    cdef:
        double loss = 0
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()
        np.ndarray[float_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        size_t N = y_true.shape[0]
        size_t n_ones = np.sum(y_true)
        size_t n_zeroes = N - n_ones

    loss = sigmoid_pairwise_loss_auc_exact_cpu(&y_true[0], &exp_pred[0], &y_pred_argsorted[0], eps, n_ones, n_zeroes, N)

    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess_auc_cpu_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                                          np.ndarray[float_t, ndim=1, mode='c'] y_pred):
    cdef:
        pair[double*, double*] grad_hess
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()
        np.ndarray[float_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        size_t N = y_true.shape[0]
        size_t n_ones = np.sum(y_true)
        size_t n_zeroes = N - n_ones

    grad_hess = sigmoid_pairwise_diff_hess_auc_cpu(&y_true[0], &exp_pred[0], &y_pred_argsorted[0], n_ones, n_zeroes, N)

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.float64_t, ndim=1, mode='c'] grad = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, <void*>grad_hess.first)
        np.ndarray[np.float64_t, ndim=1, mode='c'] hess = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, <void*>grad_hess.second)

    return grad, hess


@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess_auc_exact_cpu_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                                                np.ndarray[float_t, ndim=1, mode='c'] y_pred,
                                                double eps = 0):
    cdef:
        pair[double*, double*] grad_hess
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()
        np.ndarray[float_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        size_t N = y_true.shape[0]
        size_t n_ones = np.sum(y_true)
        size_t n_zeroes = N - n_ones
    
    grad_hess = sigmoid_pairwise_diff_hess_auc_exact_cpu(&y_true[0], &exp_pred[0], &y_pred_argsorted[0], eps, n_ones, n_zeroes, N)

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.float64_t, ndim=1, mode='c'] grad = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, <void*>grad_hess.first)
        np.ndarray[np.float64_t, ndim=1, mode='c'] hess = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, <void*>grad_hess.second)

    return grad, hess